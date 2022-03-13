from functools import singledispatchmethod

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
            self.__append_layers(Neuron(size, layers_sizes[i+1]))

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
    def weights(self) -> list:
        weight_list = []
        for i in self.layers:
            weight_list.append(i.weights.tolist())
        return weight_list

    @weights.setter
    def weights(self, value):
        self.layers = np.array([])
        for weight in value:
            self.__append_layers(Neuron(weights=np.array(weight)))

    @property
    def activation_funcs(self) -> list:
        '''List with names of activation functions'''
        return [i.activate.__name__ for i in self.layers]

    @activation_funcs.setter
    def activation_funcs(self, value: tuple):
        '''
        Sets activation functions to layers

        Gets tuple: (function for hidden layers, function for output layer)
        '''
        for i in self.layers[:-1]:
            i.activate = value[0]
        if len(value) > 1:
            self.layers[-1].activate = value[1]

    @property
    def size(self):
        '''
        Tuple with dimensions of neural network

        (input size, ... , output size)
        '''
        return (self.layers[0].weights.shape[0], *[i.weights.shape[1] for i in self.layers])

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
            layer.change_weights(delta, self.learning_rate)
            if len(layers) != 1:
                activation_func = layers[1].activate
                error = np.dot(delta, layer.weights.T)
                next_delta = error * activation_func(layer.values, True)
                return set_delta(next_delta, layers[1:])

        activation_func = self.layers[-1].activate
        output = self.forward(input_set)
        output_error = np.array(predicted_output) - output
        output_delta = output_error * activation_func(output, True)
        set_delta(output_delta, np.flip(self.layers))
        return output_error

    def train(self, epochs: int, input_set: list, predicted_outputs: list, learning_rate: float = 1) -> np.ndarray:
        '''
        Trains neural network

        Returns NDArray with error values per iteration
        '''
        self.learning_rate = learning_rate
        error_per_iteration = []
        for iter in range(epochs):
            error = self.__backward(input_set, predicted_outputs)
            error_per_iteration.append(np.average(np.abs(error)))
        return np.array(error_per_iteration)

    def forward(self, input_set: list) -> np.ndarray:
        '''Returns Numpy array with output of forward propagation'''
        return self.__calc_layer_values(np.array(input_set), self.layers)

    def initialize_weights(self, seed: int = None):
        '''
        Initialize weights of all layers according to activation functions
        
        seed: a seed to initialize weights (Must be convertible to 32 bit unsigned integers)
        '''
        np.random.seed(seed)
        for layer in self.layers:
            layer.initialize_weights()
