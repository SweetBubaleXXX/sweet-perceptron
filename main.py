from typing import Optional

import numpy as np
from multipledispatch import dispatch

np.random.seed(1)


class Neuron:
    '''Layer of neurons'''

    def __init__(self, weights):
        self.weights = weights

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return np.exp(-x) / (np.exp(-x) + 1)**2
        return 1 / (1 + np.exp(-x))

    def think(self, input_set: list) -> np.ndarray:
        '''Returns product of input and weights'''
        total = np.dot(input_set, self.weights)
        return self.sigmoid(total)


class NeuralNetwork:
    '''
    NeuralNetwork(input_size: int, neurons_per_layer: int, hidden_layers: int, output_size: int)

    NeuralNetwork(layers_sizes: list)
    '''
    @dispatch(int, int, int, int)
    def __init__(self, input_size, neurons_per_layer, hidden_layers, output_size):
        self.__append_layers(
            self.__new_layer(input_size, neurons_per_layer))
        for i in range(hidden_layers):
            self.__append_layers(self.__new_layer(
                neurons_per_layer, neurons_per_layer))
        self.__append_layers(
            self.__new_layer(neurons_per_layer, output_size))

    @dispatch(list)
    def __init__(self, layers_sizes):
        for i, size in enumerate(layers_sizes):
            if i == len(layers_sizes) - 1:
                break
            self.__append_layers(self.__new_layer(size, layers_sizes[i+1]))

    @property
    def layers(self):
        if not hasattr(self, '_layers'):
            self._layers = np.array([])
        return self._layers

    @layers.setter
    def layers(self, value) -> np.ndarray:
        self._layers = value

    def __get_random_weights(self, input: int, output: int):
        return 2 * np.random.random((input, output)) - 1

    def __new_layer(self, input_count: Optional[int], output_count: Optional[int], **kwargs):
        if not kwargs.get('weights'):
            weights = self.__get_random_weights(input_count, output_count)
        neuron = Neuron(weights)
        return neuron

    def __append_layers(self, layer):
        self.layers = np.append(self.layers, layer)

    def __calc_layer_values(self, input_set, layers):
        '''
        Calculates input and weights

        Returns value of outer layer
        '''
        layer = layers[0]
        values = layer.think(input_set)
        layer.values = values
        if len(layers) == 1:
            return values
        return self.__calc_layer_values(values, layers[1:])

    def backward(self, input_set: list, predicted_output: list):

        def set_delta(prev_delta, layers):  # move it to Neuron
            layer = layers[0]
            error = np.dot(prev_delta, layer.weights.T)
            delta = error * Neuron.sigmoid(layer.values, True)  # bug shapes
            layer.delta = delta
            if len(layers) == 1:
                return delta
            return set_delta(delta, layers[1:])

        self.forward(input_set)
        outer_layer = self.layers[-1]
        output_error = np.array(predicted_output) - outer_layer.values
        output_delta = output_error * Neuron.sigmoid(outer_layer.values, True)
        outer_layer.delta = output_delta
        set_delta(output_delta, np.flip(self.layers))

    def forward(self, input_set: list) -> np.ndarray:
        '''Returns Numpy array with output of forward propagation'''
        return self.__calc_layer_values(input_set, self.layers)

        # @property
        # def input_size(self):
        #     pass

        # @input_size.setter
        # def input_size(self, value):
        #     self._input_size = value


if __name__ == "__main__":
    nw = NeuralNetwork([4, 3, 2])
    nw.backward([1, 0, 1, 1], [1, 0])
