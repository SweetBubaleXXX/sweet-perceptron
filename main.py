import numpy as np

np.random.seed(1)


class Neuron:
    def __init__(self, weights):
        self.weights = weights

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def think(self, input_set):
        total = np.dot(self.weights, input_set)
        return self.__sigmoid(total)


class NeuralNetwork:
    def __init__(self, input_size: int = 2, neurons_per_layer: int = 3, hidden_layers: int = 1, output_size: int = 1):
        self.layers = np.array([])
        self.__append_layers(
            self.__new_layer(input_size, neurons_per_layer))
        for i in range(hidden_layers):
            self.__append_layers(self.__new_layer(
                neurons_per_layer, neurons_per_layer))
        self.__append_layers(
            self.__new_layer(neurons_per_layer, output_size))

    def __get_random_weights(self, input: int, output: int):
        return 2 * np.random.random((output, input)) - 1

    def __new_layer(self, input_count=None, output_count=None, **kwargs):
        if not kwargs.get('weights'):
            weights = self.__get_random_weights(input_count, output_count)
        neuron = Neuron(weights)
        return neuron

    def __append_layers(self, layer):
        self.layers = np.append(self.layers, layer)

    def __calc_layer_values(self, input_set, layers):
        values = layers[0].think(input_set)
        if len(layers) == 1:
            return values
        return self.__calc_layer_values(values, layers[1:])

    def forward(self, input_set):
        return self.__calc_layer_values(input_set, self.layers)

        # @property
        # def input_size(self):
        #     pass

        # @input_size.setter
        # def input_size(self, value):
        #     self._input_size = value


if __name__ == "__main__":
    nw = NeuralNetwork(3, 4, 2, 1)
