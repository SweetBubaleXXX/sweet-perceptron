import numpy as np


class Neuron:
    '''Layer of neurons'''

    def __init__(self, weights):
        self.weights = weights

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return np.exp(-x) / (np.exp(-x) + 1)**2
        return 1 / (1 + np.exp(-x))

    def change_weights(self, delta):
        '''Changes weight values according to delta'''
        self.weights += np.dot(self.values.T, delta)

    def think(self, input_set: list) -> np.ndarray:
        '''Returns product of input and weights'''
        total = np.dot(input_set, self.weights)
        return self.sigmoid(total)
