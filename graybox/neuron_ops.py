""" This module contains the interface for neuron-wise operations. """
from typing import List


class NeuronWiseOperations:
    """ This module contains the interface for neuron-wise operations. """
    MODULE_ID: int = 0

    def assign_id(self):
        """Self assign an id for each unique instance of this class."""

        self.module_id = NeuronWiseOperations.MODULE_ID
        NeuronWiseOperations.MODULE_ID += 1

    def get_module_id(self):
        """Return the id of the instance."""
        return self.module_id

    def reorder(self, indices: List[int]):
        """
        This function reorders the neurons in the layer.

        Args:
            indices: The new order of the neurons.
        """

    def prune(self, indices: List[int]):
        """
        This function prunes the neurons in the layer.

        Args:
            indices: The indices of the neurons to prune.
        """

    def reset(
            self,
            indices: List[int],
            skip_initialization: bool = False,
            perturbation_ratio: float | None = None):
        """
        This function reinitialize neurons in the layer.

        Args:
            indices: The indices of the neurons to reset.
            skip_initialization: Whether to skip the initialization step.
            perturbation_ratio: The ratio of perturbation to apply.
        """

    def add_neurons(self, neuron_count: int):
        """
        This function adds new neurons to the layer.

        Args:
            neuron_count: The number of neurons to add.
        """
