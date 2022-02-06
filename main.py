from typing import Optional

import numpy as np
from multipledispatch import dispatch


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


class NeuralNetwork:
    '''
    NeuralNetwork(input_size: int, neurons_per_layer: int, hidden_layers: int, output_size: int)

    NeuralNetwork(layers_sizes: list)
    '''
    @dispatch(int, int, int, int)  # try to replace with @overload
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
    def layers(self) -> np.ndarray:
        if not hasattr(self, '_layers'):
            self._layers = np.array([])
        return self._layers

    @layers.setter
    def layers(self, value):
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
        layer.values = input_set
        values = layer.think(input_set)
        if len(layers) == 1:
            return values
        return self.__calc_layer_values(values, layers[1:])

    def __backward(self, input_set: list, predicted_output: list):
        '''
        Calculates error and updates weights according to delta
        '''
        def set_delta(delta, layers):
            layer = layers[0]
            layer.change_weights(delta)
            if len(layers) != 1:
                error = np.dot(delta, layer.weights.T)
                next_delta = error * Neuron.sigmoid(layer.values, True)
                return set_delta(next_delta, layers[1:])

        output = self.forward(input_set)
        output_error = np.array(predicted_output) - output
        output_delta = output_error * Neuron.sigmoid(output, True)
        set_delta(output_delta, np.flip(self.layers))
        return output_error

    def train(self, epochs: int, input_set: list, predicted_outputs: list):
        error_per_iteration = np.array([])
        for iter in range(epochs):
            error = self.__backward(input_set, predicted_outputs)
            error_per_iteration = np.append(
                error_per_iteration, np.average(np.abs(error)))
        return error_per_iteration

    def forward(self, input_set: list) -> np.ndarray:
        '''Returns Numpy array with output of forward propagation'''
        return self.__calc_layer_values(np.array(input_set), self.layers)


if __name__ == "__main__":
    nw = NeuralNetwork([3, 4, 1])
    print(nw.forward([1, 0, 1]))
    inputs = [[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0],
              [0, 1, 1], [0, 0, 0], [0, 1, 0], [0, 0, 1]]
    outputs = [[i] for i in [1, 1, 1, 1, 0, 0, 0, 0]]
    error = nw.train(1000, inputs, outputs)

# set/get weights