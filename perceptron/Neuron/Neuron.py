from typing import Callable, Optional

import numpy as np

from .functions import sigmoid, xavier_init, initializations


class Neuron:
    '''
    Layer of neurons

    Default activation function is 'Sigmoid'
    '''

    def __init__(self, input_size: Optional[int] = None, output_size: Optional[int] = None, weights=None):
        if weights is None:
            weights = xavier_init(input_size, output_size)
        self.weights = weights
        self.activate: Callable = sigmoid

    def initialize_weights(self, init_func: Optional[Callable] = None):
        init_func = init_func or initializations[self.activate.__name__]
        self.weights = init_func(*self.weights.shape)

    def change_weights(self, delta, learning_rate: float):
        '''Changes weight values according to delta'''
        self.weights += np.dot(self.values.T, delta) * learning_rate

    def think(self, input_set: list) -> np.ndarray:
        '''Returns product of input and weights'''
        total = np.dot(input_set, self.weights)
        return self.activate(total)
