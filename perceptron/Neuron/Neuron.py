from typing import Callable

import numpy as np

from .functions import sigmoid


class Neuron:
    '''
    Layer of neurons

    Default activation function is 'Sigmoid'
    '''

    def __init__(self, weights):
        self.weights = weights
        self.activate: Callable  = sigmoid

    def change_weights(self, delta):
        '''Changes weight values according to delta'''
        self.weights += np.dot(self.values.T, delta)

    def think(self, input_set: list) -> np.ndarray:
        '''Returns product of input and weights'''
        total = np.dot(input_set, self.weights)
        return self.activate(total)
