from typing import Callable, Optional

import numpy as np

from .functions import sigmoid


class Neuron:
    '''
    Layer of neurons.

    Default activation function is 'Sigmoid'.

    Parameters
    ----------
    input_size : int, optional
        Number of inputs for this layer.
    output_size : int, optional
        Amount of neurons on this layer.
    weights : NDArray
        Matrix with weights of neurons.

    Attributes
    ----------
    weights : NDArray
        Matrix with weights of neurons.
    activate: Callable
        Activation function of this layer.
    '''

    def __init__(self, input_size: Optional[int] = None,
                 output_size: Optional[int] = None, weights: Optional[np.ndarray] = None):
        if weights is None:
            weights = sigmoid.__initialization__(input_size, output_size)
        self.weights = weights
        self.activate: Callable = sigmoid

    def initialize_weights(self, init_func: Optional[Callable] = None):
        """
        Initializes weights of layer depending on current activation function.
        You can pass particular initialization function.
        """
        init_func = init_func or self.activate.__initialization__
        self.weights = init_func(*self.weights.shape)

    def change_weights(self, delta, learning_rate: float):
        '''Changes weight values according to delta.'''
        self.weights += np.dot(self.values.T, delta) * learning_rate

    def think(self, input_set: list) -> np.ndarray:
        '''Returns product of input and weights.'''
        total = np.dot(input_set, self.weights)
        return self.activate(total)
