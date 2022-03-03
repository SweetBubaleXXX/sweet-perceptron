from functools import singledispatchmethod
from typing import Optional

import numpy as np

from ..Neuron import Neuron


class NeuralNetwork:
    '''
    NeuralNetwork(layers_sizes: tuple): creates neural network with given sizes

    NeuralNetwork(weight_list: list[list]): creates neural network with given set of weights
    '''

    @singledispatchmethod
    def __init__(self, arg):
        raise TypeError(f"Unknown argument type ({type(arg)})")

    @__init__.register
    def _(self, layers_sizes: tuple):
        for i, size in enumerate(layers_sizes):
            if i == len(layers_sizes) - 1:
                break
            self.__append_layers(self.__new_layer(size, layers_sizes[i+1]))

    @__init__.register
    def _(self, weight_list: list):
        self.weights = weight_list

    @property
    def layers(self) -> np.ndarray:
        if not hasattr(self, '_layers'):
            self._layers = np.array([])
        return self._layers

    @layers.setter
    def layers(self, value):
        self._layers = value

    @property
    def weights(self):
        weight_list = []
        for i in self.layers:
            weight_list.append(i.weights.tolist())
        return weight_list

    @weights.setter
    def weights(self, value):
        self.layers = np.array([])
        for weight in value:
            self.__append_layers(self.__new_layer(weights=np.array(weight)))

    def __get_random_weights(self, input: int, output: int):
        return 2 * np.random.random((input, output)) - 1

    def __new_layer(self, input_count: Optional[int] = None, output_count: Optional[int] = None, **kwargs):
        if "weights" in kwargs:
            weights = kwargs.get('weights')
        else:
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
