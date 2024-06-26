""" This module contains the interface for neuron-wise operations. """
from typing import List


class NeuronWiseOperations:
    """ This module contains the interface for neuron-wise operations. """

    def reorder(self, indices: List[int]):
        """ This function reorders the neurons in the layer. """

    def prune(self, indices: List[int]):
        """ This function prunes the neurons in the layer. """

    def reset(self, indices: List[int]):
        """ This function reinitialize neurons in the layer. """

    def add_neurons(self, neuron_count: int):
        """ This function adds new neurons to the layer. """
