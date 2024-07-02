""" Classes related to network architecture operations and internals. """
import collections
import enum

from typing import Dict, List, Set
from torch import nn

import numpy as np
import torch as th

from graybox.tracking import TrackingMode
from graybox.modules_with_ops import is_module_with_ops


class DepType(str, enum.Enum):
    """E.g: layer1.prune triggers layer2.prune_incoming."""
    INCOMING = "INCOMING"
    """E.g: layer1.insert triggers layer2.insert."""
    SAME = "SAME"
    NONE = "NONE"


class _ModulesDependencyManager:
    """
    Instead of awkwardly checking indexes in order to update dependent
    layer, we are keeping them into a dictionary in order to quickly
    look them up.
    """

    def __init__(self) -> None:
        # layer_id -> layer_ref
        self.id_2_layer = collections.defaultdict(lambda: None)
        # what kind of dependency is there
        self.dependency_2_id_2_id = collections.defaultdict(
            lambda: collections.defaultdict(lambda: []))

    def __str__(self):
        return \
            f"ModulesDependencyManager: " + \
            f"{self.id_2_layer} {self.dependency_2_id_2_id}"

    def register_module(self, id: int, module: nn.Module):
        """Register the model submodules.

        Args:
            id (int): The id of the module.
            module (nn.Module): The module associated with the id.
        """
        self.id_2_layer[id] = module

    def _register_dependency(
            self, id1: int, id2: int,
            dep_type: DepType = DepType.NONE):
        self.dependency_2_id_2_id[dep_type][id1].append(id2)

    def register_same_dependency(self, id1, id2):
        """Marks the dependency between two modules with id1 and id2 as SAME,
        in the sense that id1.operation1 triggers id2.operation1. Useful for
        when after a Linear there is a BatchNorm so adding neurons in the first
        layer triggers adding neuron the second layer.

        Args:
            id1 (int): The id of the module being depended on.
            id2 (int): The id of the module dependent on first module.
        """
        self._register_dependency(id1, id2, DepType.SAME)

    def register_incoming_dependency(self, id1, id2):
        """Marks the dependency between two modules with id1 and id2 as
        INCOMING. Useful for when after a Linear there is a Linear so adding
        neurons in the first layer triggers adding incoming neurons the second
        layer.

        Args:
            id1 (int): The id of the module being depended on.
            id2 (int): The id of the module dependent on first module.
        """
        self._register_dependency(id1, id2, DepType.INCOMING)

    def get_dependent_ids(self, idd: int, dep_type: DepType):
        """Get the ids of the modules that are dependent on the module with the
        given id.

        Args:
            id (int): The id of the module.
            dep_type (DependencyType): The type of dependency.

        Returns:
            List[int]: The ids of the dependent modules.
        """
        return list(self.dependency_2_id_2_id[dep_type][idd])

    def get_registered_ids(self) -> Set[int]:
        """Get the ids of the registered modules.

        Returns:
            List[int]: The ids of the registered modules.
        """
        return list(self.id_2_layer.keys())


def get_children(module: nn.Module):
    flatt_children = [module] if is_module_with_ops(module) else []

    if list(module.children()) == []:
        return flatt_children

    for child in module.children():
        flatt_children.extend(get_children(child))
    return flatt_children


class NetworkWithOps(nn.Module):
    def __init__(self):
        super(NetworkWithOps, self).__init__()
        self.seen_samples = 0
        self.tracking_mode = TrackingMode.DISABLED
        self._architecture_change_hook_fns = []
        self._dep_manager = _ModulesDependencyManager()

    def register_dependencies(self, dendencies_list: List):
        """Register the dependencies between children modules.

        Args:
            dependencies_dict (Dict): a dictionary in which the key is a
                pair of modules and the value is the type of the dependency
                between them.
        """
        for child_module in self.layers:
            self._dep_manager.register_module(
                id(child_module), child_module)

        for module1, module2, value in dendencies_list:
            id1, id2 = id(module1), id(module2)
            if value == DepType.INCOMING:
                self._dep_manager.register_incoming_dependency(id1, id2)
            elif value == DepType.SAME:
                self._dep_manager.register_same_dependency(id1, id2)

    @property
    def layers(self):
        return get_children(self)

    def reset_stats(self):
        for layer in self.layers:
            if hasattr(layer, "reset_stats"):
                layer.reset_stats()

    def get_parameter_count(self):
        count = 0
        for layer in self.parameters():
            count += np.prod(layer.shape)
        return count

    def register_hook_fn_for_architecture_change(self, fn):
        self._architecture_change_hook_fns.append(fn)

    def __hash__(self):
        return hash(str(self))

    def set_tracking_mode(self, mode: TrackingMode):
        self.tracking_mode = mode
        for layer in self.layers:
            layer.tracking_mode = mode

    def to(self, device, dtype=None, non_blocking=False, **kwargs):
        self.device = device
        super().to(device, dtype, non_blocking, **kwargs)
        for layer in self.layers:
            layer.to(device, dtype, non_blocking, **kwargs)

    def maybe_update_age(self, tracked_input: th.Tensor):
        if self.tracking_mode != TrackingMode.TRAIN:
            return
        if not hasattr(tracked_input, 'batch_size'):
            setattr(tracked_input, 'batch_size', tracked_input.shape[0])
        self.seen_samples += tracked_input.batch_size

    def get_age(self):
        return self.seen_samples

    def reinit_neurons(
            self,
            layer_id: int,
            neuron_indices: Set[int],
            perturbation_ratio: float | None = None):
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.prune] No module with id {layer_id}")

        module = self._dep_manager.id_2_layer[layer_id]
        module.reset(neuron_indices, perturbation_ratio=perturbation_ratio)

        # If the dependent layer is of type "SAME", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        for same_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.SAME):
            self.reinit_neurons(
                same_dep_id, neuron_indices, perturbation_ratio)

        # If the next layer is of type "INCOMING", say after a conv we have 
        # either a conv or a linear, then we add to incoming neurons.
        for incoming_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.INCOMING):
            incoming_module = self._dep_manager.id_2_layer[incoming_id]
            incoming_module.reset_incoming_neurons(
                neuron_indices,
                skip_initialization=True,
                perturbation_ratio=perturbation_ratio)

        # TODO(rotaru): Deal with through_flatten case.

    def _conv_neuron_to_linear_neurons_through_flatten(
            self, conv_layer, linear_layer):
        conv_neurons = conv_layer.weight.shape[0]
        linear_neurons = linear_layer.weight.shape[1]
        linear_neurons_per_conv_neuron = linear_neurons // conv_neurons
        return linear_neurons_per_conv_neuron

    def prune(
            self,
            layer_id: int,
            neuron_indices: Set[int],
            through_flatten: bool = False):

        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.prune] No module with id {layer_id}")

        module = self._dep_manager.id_2_layer[layer_id]
        module.prune(neuron_indices)

        # If the dependent layer is of type "SAME", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        for same_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.SAME):
            self.prune(same_dep_id, neuron_indices)

        # If the next layer is of type "INCOMING", say after a conv we have 
        # either a conv or a linear, then we add to incoming neurons.
        for incoming_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.INCOMING):
            incoming_module = self._dep_manager.id_2_layer[incoming_id]
            incoming_module.prune_incoming_neurons(neuron_indices)

        # TODO(rotaru): Deal with through_flatten case.

        for hook_fn in self._architecture_change_hook_fns:
            hook_fn(self)

        # self.spike_scalars["prune_neurons"].value = 1.0

    def add_neurons(self,
                    layer_id: int,
                    neuron_count: int,
                    skip_initialization: bool = True,
                    through_flatten: bool = False):

        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.add_neurons] No module with id {layer_id}")

        module = self._dep_manager.id_2_layer[layer_id]
        module.add_neurons(
                neuron_count, skip_initialization=skip_initialization)

        # If the dependent layer is of type "SAME", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        for same_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.SAME):
            self.add_neurons(same_dep_id, neuron_count, skip_initialization)

        # If the next layer is of type "INCOMING", say after a conv we have 
        # either a conv or a linear, then we add to incoming neurons.
        for incoming_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.INCOMING):
            incoming_module = self._dep_manager.id_2_layer[incoming_id]
            incoming_module.add_incoming_neurons(
                neuron_count, skip_initialization)

        # TODO(rotaru): Deal with through_flatten case.

        for hook_fn in self._architecture_change_hook_fns:
            hook_fn(self)

    def reorder(self,
                layer_id: int,
                indices: List[int],
                through_flatten: bool = False):

        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.reorder] No module with id {layer_id}")

        module = self._dep_manager.id_2_layer[layer_id]
        module.reorder(indices)

        # If the dependent layer is of type "SAME", say after a conv we have
        # batch_norm, then we have to update the layer after the batch_norm too
        for same_dep_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.SAME):
            self.reorder(same_dep_id, indices)

        # If the next layer is of type "INCOMING", say after a conv we have 
        # either a conv or a linear, then we add to incoming neurons.
        for incoming_id in self._dep_manager.get_dependent_ids(
                layer_id, DepType.INCOMING):
            incoming_module = self._dep_manager.id_2_layer[incoming_id]
            incoming_module.reorder_incoming_neurons(indices)

        # TODO(rotaru): Deal with through_flatten case.

    def reorder_neurons_by_trigger_rate(self, layer_id: int):
        if layer_id not in self._dep_manager.id_2_layer:
            raise ValueError(
                f"[NetworkWithOps.reorder_by] No module with id {layer_id}")

        module = self._dep_manager.id_2_layer[layer_id]
        if not hasattr(module, 'train_dataset_tracker'):
            raise ValueError(
                f"[NetworkWithOps.reorder_by] Module with id {layer_id} "
                f"has not trackers")
        tracker = module.train_dataset_tracker

        ids_and_rates = []
        for neuron_id in range(tracker.number_of_neurons):
            frq_curr = tracker.get_neuron_stats(neuron_id)
            ids_and_rates.append((neuron_id, frq_curr))
        ids_and_rates.sort(key=lambda x: x[1], reverse=True)
        indices = [idx_and_frq[0] for idx_and_frq in ids_and_rates]

        self.reorder(layer_id=layer_id, indices=indices)

    def freeze_layers_up_to_idx(self, last_frozen_layer_idx: int):
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx <= last_frozen_layer_idx:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_layers_from_idx(self, first_unfrozen_layer_idx: int):
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx >= first_unfrozen_layer_idx:
                for param in layer.parameters():
                    param.requires_grad = True

    def model_summary_str(self):
        repr = "Model|"
        for layer in self.layers:
            repr += layer.summary_repr() + "|"
        return repr

    def __eq__(self, other: "NetworkWithOps") -> bool:
        return self.seen_samples == other.seen_samples and \
            self.tracking_mode == other.tracking_mode and \
            self.layers == other.layers

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        state[prefix + 'seen_samples'] = self.seen_samples
        state[prefix + 'tracking_mode'] = self.tracking_mode
        return state

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        self.seen_samples = state_dict[prefix + 'seen_samples']
        self.tracking_mode = state_dict[prefix + 'tracking_mode']
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def __repr__(self):
        return super().__repr__() + f" age=({self.seen_samples})"

    def forward(self,
                tensor: th.Tensor,
                intermediary_outputs: List[int] = []):
        raise NotImplementedError
