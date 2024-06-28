""" Wrappers over common pytorch modules that allow for interactive changes. """
import collections

from typing import Set, List, Union
from torch import nn
import torch as th

from graybox.neuron_ops import NeuronWiseOperations
from graybox.tracking import TrackingMode
from graybox.tracking import Tracker
from graybox.tracking import TriggersTracker
from graybox.tracking import copy_forward_tracked_attrs


# TODO(rotaru): Apply ops on gradient tensors too.


class LayerWiseOperations(NeuronWiseOperations):
    """
        Base class for the complementary operations needed in order to
        implement the neuron wise operations correctly.
    """
    object_counter: int = 0

    def __init__(
            self,
            neuron_count: int,
            incoming_neuron_count: int,
            device,
            module_name: str = "module") -> None:
        super().__init__()
        self.id = LayerWiseOperations.object_counter
        LayerWiseOperations.object_counter += 1
        self.neuron_count = neuron_count
        self.incoming_neuron_count = incoming_neuron_count
        self.device = device
        self.tracking_mode = TrackingMode.DISABLED
        self.neuron_2_learning_rate = collections.defaultdict(lambda: 1.0)
        self.incoming_neuron_2_lr = collections.defaultdict(lambda: 1.0)

    def set_tracking_mode(self, tracking_mode: TrackingMode):
        """ Set what samples are the stats related to (train/eval/etc). """
        self.tracking_mode = tracking_mode

    def reset_incoming_neurons(
            self,
            indices: Set[int],
            skip_initialization: bool = False,
            perturbation_ratio: float | None = None):
        """
            When re-initializing a neuron (a.k.a a line or kernel in a layer)
            we need to apply the operation on the counter part in the next
            layer.
            Parameters
            ----------
            indices : Set[int]
                The list of indices of the incoming neurons placed in the
                position where they should appear in resulting tensor.
            skip_initialization : bool, optional
                Weather to apply standard initialization of the re-initialized
                neurons.
            perturbation_ratio : flat, optional
                If not None, the neurons will be perturbed with slight noise
                within the  perturbation_ratio percent from the actual value.
                new_values ~ [
                    old_value * (1 - perturbation_ratio),
                    old_value * (1 + perturbation_ratio)
                ]
        """
        raise NotImplementedError

    def reorder_incoming_neurons(self, indices: Set[int]):
        """
        This method allows you to reorder the incoming neurons in the current
        layer based on the given indices.

        Parameters:
            indices (Set[int]):
                A set of integers representing the new order of the incoming
                neurons.
        """
        raise NotImplementedError

    def prune_incoming_neurons(self, indices: Set[int]):
        """
        This method allows you to prune the incoming neurons in the current
        layer.

        Args:
            indices (Set[int]):
                The indices of the incoming neurons to be pruned.
        """
        raise NotImplementedError

    def add_incoming_neurons(
            self,
            neuron_count: int,
            skip_initialization: bool = False):
        """Add incoming neurons to the current layer.

        Args:
            neuron_count (int): the number of neurons to be added.
            skip_initialization (bool, optional):
                whether to skip standard initialization. Defaults to False.
        """
        raise NotImplementedError

    def set_per_neuron_learning_rate(self, neuron_ids: Set[int], lr: float):
        """
        Set per neuron learning rates.

        Args:
            neuron_ids (Set[int]): The set of neurons to set the learning rate
            lr (float): The value of the learning rate. Can be between [0, 1]
        """
        if lr < 0 or lr > 1.0:
            raise ValueError('Cannot set learning rate outside [0, 1] range')

        invalid_ids = (neuron_ids - set(range(self.neuron_count)))
        if invalid_ids:
            raise ValueError(
                f'Cannot set learning rate for neurons {invalid_ids} as they '
                f'are outside the set of existent neurons {self.neuron_count}.'
            )

        for neuron_id in neuron_ids:
            self.neuron_2_learning_rate[neuron_id] = lr

    def set_per_incoming_neuron_learning_rate(
            self, neuron_ids: Set[int], lr: float):
        """
        Set learning rate per incoming neuron.

        Args:
            neuron_ids (Set[int]):
                The set of incoming neurons to set the learning rate
            lr (float): The value of the learning rate. Can be between [0, 1]
        """

        if lr < 0 or lr > 1.0:
            raise ValueError('Cannot set learning rate outside [0, 1] range')

        invalid_ids = (neuron_ids - set(range(self.incoming_neuron_count)))
        if invalid_ids:
            raise ValueError(
                f'Cannot set learning rate for neurons {invalid_ids} as they '
                f'are outside the set of existent neurons '
                f'{self.incoming_neuron_count}.'
            )

        for neuron_id in neuron_ids:
            self.incoming_neuron_2_lr[neuron_id] = lr

    def register_grad_hook(self):
        # This is meant to be called in the children classes.
        def weight_grad_hook(weight_grad):
            for neuron_id, neuron_lr in self.neuron_2_learning_rate.items():
                neuron_grad = weight_grad[neuron_id]
                neuron_grad *= neuron_lr
                weight_grad[neuron_id] = neuron_grad

            for in_neuron_id, neuron_lr in self.incoming_neuron_2_lr.items():
                in_neuron_grad = weight_grad[:, in_neuron_id]
                in_neuron_grad *= neuron_lr
                weight_grad[:, in_neuron_id] = in_neuron_grad
            return weight_grad

        def bias_grad_hook(bias_grad):
            for neuron_id, neuron_lr in self.neuron_2_learning_rate.items():
                neuron_grad = bias_grad[neuron_id]
                neuron_grad *= neuron_lr
                bias_grad[neuron_id] = neuron_grad
            return bias_grad

        if hasattr(self, 'weight'):
            self.weight.register_hook(weight_grad_hook)
        if hasattr(self, 'bias'):
            self.bias.register_hook(bias_grad_hook)

    def _find_value_for_key_pattern(self, key_pattern, state_dict):
        for key, value in state_dict.items():
            if key_pattern in key:
                return value
        return None


class LinearWithNeuronOps(nn.Linear, LayerWiseOperations):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype)
        LayerWiseOperations.__init__(self, out_features, in_features, device)
        self.register_module('train_dataset_tracker', TriggersTracker(
            self.neuron_count, device=self.device))
        self.register_module('eval_dataset_tracker', TriggersTracker(
            self.neuron_count, device=self.device))
        self.register_grad_hook()

    def reset_stats(self):
        """Reset stats for the trackers."""
        self.train_dataset_tracker.reset_stats()
        self.eval_dataset_tracker.reset_stats()

    def get_parameter_count(self):
        """Compute the rough number of parameters in the layer.

        Returns:
            int: The number of parameters in the layer.
        """
        parameters = 1
        for i in range(len(self.weight.shape)):
            parameters *= self.weight.shape[i]
        if self.bias:
            parameters += self.bias.shape[0]
        return parameters

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        tnsr = self._find_value_for_key_pattern('.weight', state_dict)
        if tnsr is not None:
            in_size, out_size = tnsr.shape[1], tnsr.shape[0]
            with th.no_grad():
                wshape = (out_size, in_size)
                self.weight.data = nn.Parameter(
                    th.ones(wshape)).to(self.device)
                self.bias.data = nn.Parameter(
                    th.ones(out_size)).to(self.device)
            self.in_features = in_size
            self.incoming_neuron_count = in_size
            self.out_features = out_size
            self.neuron_count = out_size
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def __eq__(self, other: "LinearWithNeuronOps") -> bool:
        return th.allclose(self.weight.data, other.weight.data) and \
            th.allclose(self.bias.data, other.bias.data) and \
            self.train_dataset_tracker == other.train_dataset_tracker and \
            self.eval_dataset_tracker == other.eval_dataset_tracker

    def __hash__(self):
        return hash(str(self))

    def get_tracker(self) -> Tracker:
        if self.tracking_mode == TrackingMode.TRAIN:
            return self.train_dataset_tracker
        elif self.tracking_mode == TrackingMode.EVAL:
            return self.eval_dataset_tracker
        else:
            return None

    def trackers(self):
        return [self.eval_dataset_tracker, self.train_dataset_tracker]

    def register(
            self,
            activation_map: th.Tensor):
        tracker = self.get_tracker()
        if tracker is None or tracker is None or activation_map is None:
            return

        activation_map_bin_ed = activation_map > 0
        copy_forward_tracked_attrs(activation_map_bin_ed, activation_map)
        tracker.update(activation_map_bin_ed)

    def to(self, device):
        self.device = device
        super().to(device)
        for tracker in self.trackers():
            tracker.to(device)

    def reorder(self, indices: List[int]):
        neurons = set(range(self.out_features))
        if not set(indices) & neurons:
            raise ValueError(
                f"LinearWithNeuronOps.reorder indices and neurons do not "
                f"overlapp: {indices} & {neurons} => {indices & neurons}")
        idx_tnsr = th.tensor(indices).to(self.device)
        with th.no_grad():
            self.weight.data = nn.Parameter(
                self.weight.data[idx_tnsr]).to(self.device)
            self.bias.data = nn.Parameter(
                self.bias.data[idx_tnsr]).to(self.device)

        for tracker in self.trackers():
            tracker.reorder(indices)

    def reorder_incoming_neurons(self, indices: List[int]):
        neurons = set(range(self.out_features))
        if not set(indices) & neurons:
            raise ValueError(
                f"LinearWithNeuronOps.reorder_incoming_neurons indices and "
                f"neurons set do not overlapp: {indices} & {neurons} => "
                f"{indices & neurons}")
        idx_tnsr = th.tensor(indices).to(self.device)
        with th.no_grad():
            self.weight.data = nn.Parameter(
                self.weight.data[:, idx_tnsr]).to(self.device)

    def prune(self, indices: Set[int]):
        neurons = set(range(self.out_features))
        if not set(indices) & neurons:
            raise ValueError(
                f"LinearWithNeuronOps.prune indices and neurons set do not "
                f"overlapp: {indices} & {neurons} => {indices & neurons}")

        kept_neurons = th.tensor(sorted(list(neurons - indices))).to(
            self.device)
        with th.no_grad():
            self.weight.data = nn.Parameter(
                self.weight.data[kept_neurons]).to(self.device)
            self.bias.data = nn.Parameter(
                self.bias.data[kept_neurons]).to(self.device)

        self.out_features = len(kept_neurons)
        self.neuron_count = self.out_features
        for tracker in self.trackers():
            tracker.prune(indices)

    def prune_incoming_neurons(self, indices: Set[int]):
        neurons = set(range(self.in_features))
        if not set(indices) & neurons:
            raise ValueError(
                f"LinearWithNeuronOps.prune_incoming_neurons indices and "
                f"neurons set do not overlapp: {indices} & {neurons} => "
                f"{indices & neurons}")

        kept_neurons = th.tensor(sorted(list(neurons - indices))).to(
            self.device)
        with th.no_grad():
            self.weight.data = nn.Parameter(
                self.weight.data[:, kept_neurons]).to(self.device)

        self.in_features = len(kept_neurons)
        self.incoming_neuron_count = self.in_features

    def reset(
            self,
            indices: Set[int],
            skip_initialization: bool = False,
            perturbation_ratio: float | None = None):
        # Skip initialization is only to be able to test the function.
        neurons = set(range(self.out_features))
        if not set(indices) & neurons:
            raise ValueError(
                f"LinearWithNeuronOps.reset indices and neurons set do not "
                f"overlapp: {indices} & {neurons} => "
                f"{indices & neurons}")
        if perturbation_ratio is not None and (
                0.0 >= perturbation_ratio or perturbation_ratio >= 1.0):
            raise ValueError(
                f"LinearWithOps.reset perturbation "
                f"{perturbation_ratio} outside of [0.0, 1.0]")

        with th.no_grad():
            for neuron_id in indices:
                neuron_weights = th.zeros(self.in_features).to(self.device)
                neuron_bias = 0.0
                if not skip_initialization:
                    neuron_weights = neuron_weights.unsqueeze(0)
                    nn.init.xavier_uniform_(
                        neuron_weights, gain=nn.init.calculate_gain('relu'))
                    neuron_weights = neuron_weights.squeeze(0)

                    if perturbation_ratio is not None:
                        neuron_weights = self.weight[neuron_id]
                        neuron_bias = self.bias[neuron_id]
                        neuron_perturb = \
                            perturbation_ratio * neuron_weights * \
                            th.randint_like(neuron_weights, -1, 2).float()
                        bias_perturb = \
                            perturbation_ratio * neuron_bias * \
                            th.randint(-1, 2, (1, )).float().item()

                        neuron_weights += neuron_perturb
                        neuron_bias += bias_perturb

                self.weight[neuron_id] = neuron_weights
                self.bias[neuron_id] = neuron_bias

        for tracker in self.trackers():
            tracker.reset(indices)

    def reset_incoming_neurons(
            self,
            indices: Set[int],
            skip_initialization: bool = False,
            perturbation_ratio: float | None = None):
        neurons = set(range(self.in_features))
        if not set(indices) & neurons:
            raise ValueError(
                f"LinearWithNeuronOps.reset_incoming_neurons indices and "
                f"neurons set do not overlapp: {indices} & {neurons} => "
                f"{indices & neurons}")

        if perturbation_ratio is not None and (
                0.0 >= perturbation_ratio or perturbation_ratio >= 1.0):
            raise ValueError(
                f"LinearWithOps.reset perturbation "
                f"{perturbation_ratio} outside of [0.0, 1.0]")

        with th.no_grad():
            column = th.zeros(self.out_features).to(self.device)
            rowidx = th.arange(self.out_features).long().to(self.device)
            for neuron_id in indices:
                colidx = (th.ones(self.out_features) * neuron_id).long().to(
                    self.device)
                if not skip_initialization:
                    column = column.unsqueeze(0)
                    nn.init.xavier_uniform_(
                        column, gain=nn.init.calculate_gain('relu'))
                    column = column.squeeze(0)
                    if perturbation_ratio is not None:
                        column = self.weight[:, neuron_id]
                        column_perturb = \
                            perturbation_ratio * column * \
                            th.randint_like(column, -1, 2).float()
                        column += column_perturb
                self.weight[rowidx, colidx] = column

    def add_neurons(self,
                    neuron_count: int,
                    skip_initialization: bool = False):
        added_bias = th.zeros(neuron_count).to(self.device)
        added_weights = th.zeros(neuron_count, self.in_features).to(
            self.device)
        if not skip_initialization:
            nn.init.xavier_uniform_(added_weights,
                                    gain=nn.init.calculate_gain('relu'))

        with th.no_grad():
            self.weight.data = nn.Parameter(
                th.cat((self.weight.data, added_weights))).to(self.device)
            self.bias.data = nn.Parameter(
                th.cat((self.bias.data, added_bias))).to(self.device)
        self.out_features += neuron_count
        self.neuron_count = self.out_features

        super().to(self.device)
        for tracker in self.trackers():
            tracker.add_neurons(neuron_count)

    def add_incoming_neurons(
            self,
            neuron_count: int,
            skip_initialization: bool = False):
        new_wght = th.zeros((self.out_features, neuron_count)).to(self.device)
        if not skip_initialization:
            nn.init.xavier_uniform_(new_wght,
                                    gain=nn.init.calculate_gain('relu'))
        self.in_features += neuron_count
        self.incoming_neuron_count = self.in_features
        with th.no_grad():
            self.weight.data = nn.Parameter(
                th.cat((self.weight.data, new_wght), dim=1)).to(self.device)
        super().to(self.device)

    def forward(self, data: th.Tensor, skip_register: bool = False):
        activation_map = super().forward(data)
        if not skip_register:
            copy_forward_tracked_attrs(activation_map, data)
            self.register(activation_map)
        return activation_map

    def summary_repr(self):
        return f"LinearWithNeuronOps[{self.in_features}->{self.out_features}]"


class Conv2dWithNeuronOps(nn.Conv2d, LayerWiseOperations):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,        
        dtype=None
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype)
        LayerWiseOperations.__init__(
            self, out_channels, in_channels, device, "conv2d")

        self.register_module(
            'train_dataset_tracker',
            TriggersTracker(self.neuron_count, device=self.device))
        self.register_module(
            'eval_dataset_tracker',
            TriggersTracker(self.neuron_count, device=self.device))
        self.register_grad_hook()

    def reset_stats(self):
        self.train_dataset_tracker.reset_stats()
        self.eval_dataset_tracker.reset_stats()

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        tnsr = self._find_value_for_key_pattern('.weight', state_dict)
        if tnsr is not None:
            in_size, out_size = tnsr.shape[1], tnsr.shape[0]
            with th.no_grad():
                wshape = (out_size, in_size, *self.kernel_size)
                self.weight.data = nn.Parameter(
                    th.ones(wshape)).to(self.device)
                if self.bias is not None:
                    self.bias.data = nn.Parameter(
                        th.ones(out_size)).to(self.device)
            self.in_channels = in_size
            self.incoming_neuron_count = in_size
            self.out_channels = out_size
            self.neuron_count = out_size
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: "LinearWithNeuronOps") -> bool:
        return self.weight.device == other.weight.device and \
            th.allclose(self.weight.data, other.weight.data) and \
            th.allclose(self.bias.data, other.bias.data) and \
            self.train_dataset_tracker == other.train_dataset_tracker and \
            self.eval_dataset_tracker == other.eval_dataset_tracker

    def get_tracker(self) -> Tracker:
        if self.tracking_mode == TrackingMode.TRAIN:
            return self.train_dataset_tracker
        elif self.tracking_mode == TrackingMode.EVAL:
            return self.eval_dataset_tracker
        else:
            return None

    def trackers(self):
        return [self.eval_dataset_tracker, self.train_dataset_tracker]

    def register(
            self,
            activation_map: th.Tensor,
            original_input: th.Tensor):
        tracker = self.get_tracker()
        if tracker is None or activation_map is None or input is None:
            return

        activation_map = (activation_map > 0).long()

        # This should work for both batched input and un-batched inputs
        # (such as in tests).
        processed_activation_map = th.sum(activation_map, dim=(-2, -1))
        copy_forward_tracked_attrs(processed_activation_map, original_input)
        tracker.update(processed_activation_map)

    def to(self, device):
        self.device = device
        super().to(device)
        for tracker in self.trackers():
            tracker.to(device)

    def reorder(self, indices: List[int]):
        neurons = set(range(self.out_channels))
        if not set(indices) & neurons:
            raise ValueError(
                f"Conv2dWithNeuronOps.reorder indices and neurons set do not "
                f"overlapp: {indices} & {neurons} => {indices & neurons}")
        idx_tnsr = th.tensor(indices).to(self.device)
        with th.no_grad():
            self.weight.data = nn.Parameter(
                th.index_select(self.weight.data, dim=0, index=idx_tnsr)).to(
                    self.device)
            self.bias.data = nn.Parameter(
                th.index_select(self.bias.data, dim=0, index=idx_tnsr)).to(
                    self.device)

        for tracker in self.trackers():
            tracker.reorder(indices)

    def reorder_incoming_neurons(self, indices: List[int]):
        neurons = set(range(self.in_channels))
        if not set(indices) & neurons:
            raise ValueError(
                f"Conv2dWithNeuronOps.reorder_incoming_neurons indices and "
                f" neurons set do not overlapp: {indices} & {neurons} => "
                f"{indices & neurons}")

        with th.no_grad():
            idx_tnsr = th.tensor(indices).to(self.device)
            self.weight.data = nn.Parameter(
                th.index_select(self.weight.data, dim=1, index=idx_tnsr)).to(
                    self.device)

    def prune(self, indices: Set[int]):
        curr_neurons = set(range(self.out_channels))
        if not set(indices) & curr_neurons:
            raise ValueError(
                f"Conv2dWithNeuronOps.prune indices and neurons set do not "
                f"overlapp: {indices} & {curr_neurons} => "
                f"{indices & curr_neurons}")
        kept_neurons = curr_neurons - indices
        idx_tnsr = th.tensor(list(curr_neurons - indices)).to(self.device)

        with th.no_grad():

            self.weight.data = nn.Parameter(
                th.index_select(self.weight.data, dim=0, index=idx_tnsr)).to(
                    self.device)
            self.bias.data = nn.Parameter(
                th.index_select(self.bias.data, dim=0, index=idx_tnsr)).to(
                    self.device)

        self.out_channels = len(kept_neurons)
        self.neuron_count = self.out_channels
        for tracker in self.trackers():
            tracker.prune(indices)

        super().to(self.device)

    def prune_incoming_neurons(self, indices: Set[int]):
        curr_neurons = set(range(self.in_channels))
        if not set(indices) & curr_neurons:
            raise ValueError(
                f"Conv2dWithNeuronOps.prune_incoming_neurons indices and "
                f"neurons set do not overlapp: {indices} & {curr_neurons} => "
                f"{indices & curr_neurons}")
        kept_neurons = curr_neurons - indices
        idx_tnsr = th.tensor(list(kept_neurons)).to(self.device)

        with th.no_grad():
            self.weight.data = nn.Parameter(
                th.index_select(self.weight.data, dim=1, index=idx_tnsr)).to(
                    self.device)
        self.in_channels = len(kept_neurons)
        self.incoming_neuron_count = self.in_channels

    def reset(
            self,
            indices: Set[int],
            skip_initialization: bool = False,
            perturbation_ratio: float | None = None):
        neurons = set(range(self.out_channels))
        if not set(indices) & neurons:
            raise ValueError(
                f"Conv2dWithNeuronOps.reset indices and neurons set do not "
                f"overlapp: {indices} & {neurons} => {indices & neurons}")

        if perturbation_ratio is not None and (
                0.0 >= perturbation_ratio or perturbation_ratio >= 1.0):
            raise ValueError(
                f"LinearWithOps.reset perturbation "
                f"{perturbation_ratio} outside of [0.0, 1.0]")

        with th.no_grad():
            for neuron_id in indices:
                neuron_weights = th.zeros(self.in_channels, *self.kernel_size).to(
                    self.device)
                neuron_bias = 0.0
                if not skip_initialization:
                    nn.init.xavier_uniform_(neuron_weights,
                                            gain=nn.init.calculate_gain('relu'))
                    if perturbation_ratio is not None:
                        neuron_weights = self.weight[neuron_id]
                        neuron_bias = self.bias[neuron_id]

                        weights_perturbation = \
                            perturbation_ratio * neuron_weights * \
                            th.randint_like(neuron_weights, -1, 2).float()
                        bias_perturbation = \
                            perturbation_ratio * neuron_bias * \
                            th.randint(-1, 2, (1, )).float().item()

                        neuron_weights += weights_perturbation
                        neuron_bias += bias_perturbation
                self.weight[neuron_id] = neuron_weights
                self.bias[neuron_id] = neuron_bias

        for tracker in self.trackers():
            tracker.reset(indices)

    def reset_incoming_neurons(
            self,
            indices: Set[int],
            skip_initialization: bool = False,
            perturbation_ratio: float | None = None):
        neurons = set(range(self.in_channels))
        if not set(indices) & neurons:
            raise ValueError(
                f"LinearWithNeuronOps.reset_incoming_neurons indices and "
                f"neurons set do not overlapp: {indices} & {neurons} => "
                f"{indices & neurons}")

        if perturbation_ratio is not None and (
                0.0 >= perturbation_ratio or perturbation_ratio >= 1.0):
            raise ValueError(
                f"LinearWithOps.reset perturbation "
                f"{perturbation_ratio} outside of [0.0, 1.0]")

        with th.no_grad():
            for neuron_id in indices:
                overriding_weights = th.zeros(
                    self.out_channels, *self.kernel_size).to(self.device)
                if not skip_initialization:
                    # TODO(rotaru): Revise this.
                    nn.init.xavier_uniform_(
                        overriding_weights,
                        gain=nn.init.calculate_gain('relu'))
                    if perturbation_ratio is not None:
                        overriding_weights = self.weight[:, neuron_id]
                        weights_perturb = perturbation_ratio * \
                            overriding_weights * \
                            th.randint_like(overriding_weights, -1, 2).float()
                        overriding_weights += weights_perturb

                self.weight[:, neuron_id] = overriding_weights

    def add_neurons(self,
                    neuron_count: int,
                    skip_initialization: bool = False):
        parameters = th.zeros(
            (neuron_count, self.in_channels, *self.kernel_size)
        ).to(self.device)
        biases = th.zeros(neuron_count,).to(self.device)
        if not skip_initialization:
            nn.init.xavier_uniform_(parameters,
                                    gain=nn.init.calculate_gain('relu'))

        with th.no_grad():
            self.weight.data = nn.Parameter(th.cat((self.weight.data,
                                                    parameters)))
            self.bias.data = nn.Parameter(th.cat((self.bias.data, biases)))

        self.out_channels += neuron_count
        self.neuron_count = self.out_channels

        for tracker in self.trackers():
            tracker.add_neurons(neuron_count)

        super().to(self.device)

    def add_incoming_neurons(
            self, neuron_count: int, skip_initialization: bool = False):
        parameters = th.zeros(
            (self.out_channels, neuron_count, *self.kernel_size)
        ).to(self.device)
        if not skip_initialization:
            nn.init.xavier_uniform_(
                parameters, gain=nn.init.calculate_gain('relu'))

        with th.no_grad():
            self.weight.data = nn.Parameter(
                th.cat((self.weight.data, parameters), dim=1))

        self.in_channels += neuron_count
        self.incoming_neuron_count = self.in_channels
        super().to(self.device)

    def forward(self,
                input: th.Tensor,
                skip_register: bool = False):
        activation_map = super().forward(input)
        if not skip_register:
            self.register(activation_map, input)
        return activation_map

    def summary_repr(self):
        return f"conv[{self.in_channels} " + \
            f"-> {self.kernel_size} " + \
            f"-> {self.out_channels}]"


class BatchNorm2dWithNeuronOps(nn.BatchNorm2d, LayerWiseOperations):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, device, dtype
        )
        self.device = device

    def to(self, device):
        super().to(device)
        self.device = device

    def reorder(self, indices: List[int]):
        neurons = set(range(self.num_features))
        if not set(indices) & neurons:
            raise ValueError(
                f"BatchNorm2dWithNeuronOps.reorder indices and neurons set do not "
                f"overlapp: {indices} & {neurons} => {indices & neurons}")
        idx_tnsr = th.tensor(indices).to(self.device)
        with th.no_grad():
            if self.weight is not None:
                self.weight.data = nn.Parameter(
                    th.index_select(self.weight.data, dim=0, index=idx_tnsr)).to(
                        self.device)
                self.bias.data = nn.Parameter(
                    th.index_select(self.bias.data, dim=0, index=idx_tnsr)).to(
                        self.device)

            self.running_mean = th.index_select(
                self.running_mean, dim=0, index=idx_tnsr)
            self.running_var = th.index_select(
                self.running_var, dim=0, index=idx_tnsr)

    def prune(self, indices: List[int]):
        curr_neurons = set(range(self.num_features))
        if not set(indices) & curr_neurons:
            raise ValueError(
                f"BatchNorm2dWithNeuronOps.prune indices and neurons set do not "
                f"overlapp: {indices} & {curr_neurons} => "
                f"{indices & curr_neurons}")
        kept_neurons = curr_neurons - indices
        idx_tnsr = th.tensor(list(curr_neurons - indices)).to(self.device)

        with th.no_grad():
            if self.weight is not None:
                self.weight.data = nn.Parameter(
                    th.index_select(self.weight.data, dim=0, index=idx_tnsr)).to(
                        self.device)
                self.bias.data = nn.Parameter(
                    th.index_select(self.bias.data, dim=0, index=idx_tnsr)).to(
                        self.device)

            self.running_mean = th.index_select(
                self.running_mean, dim=0, index=idx_tnsr)
            self.running_var = th.index_select(
                self.running_var, dim=0, index=idx_tnsr)

        self.out_channels = len(kept_neurons)
        self.neuron_count = self.out_channels

    def reset(self, indices: List[int]):
        neurons = set(range(self.num_features))
        if not set(indices) & neurons:
            raise ValueError(
                f"BatchNorm2dWithNeuronOps.reset indices and neurons set do not "
                f"overlapp: {indices} & {neurons} => {indices & neurons}")

        for neuron_id in indices:
            self.weight[neuron_id] = th.zeros_like(self.weight[neuron_id])
            self.bias[neuron_id] = 0
            self.running_mean[neuron_id] = 0.0
            self.running_var[neuron_id] = 1.0

    def add_neurons(self, neuron_count: int):
        parameters = th.ones(neuron_count, ).to(self.weight.device)
        biases = th.zeros(neuron_count,).to(self.weight.device)
        with th.no_grad():
            self.weight.data = nn.Parameter(th.cat((self.weight.data,
                                                    parameters)))
            self.bias.data = nn.Parameter(th.cat((self.bias.data, biases)))

            self.running_mean = th.cat((
                self.running_mean,
                th.zeros(neuron_count).to(self.running_mean.device)))
            self.running_var = th.cat((
                self.running_var,
                th.ones(neuron_count).to(self.running_var.device))
            )
        self.num_features += neuron_count


def is_module_with_ops(obj):
    return issubclass(type(obj), LayerWiseOperations)
