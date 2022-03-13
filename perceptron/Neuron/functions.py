from numpy import exp
import numpy as np


def linear(x, derivative=False):
    return x


def sigmoid(x, derivative=False):
    if derivative:
        return exp(-x) / (exp(-x) + 1)**2
    return 1 / (1 + exp(-x))


def tanh(x, derivative=False):
    if derivative:
        return 4 * exp(2 * x) / (exp(2 * x) + 1)**2
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(int)
    return np.maximum(0, x)


def softmax(x):
    '''Note: can't use in neural network because of no derivative'''
    return exp(x) / np.sum(exp(x))


def xavier_init(input: int, output: int):
    scale = np.sqrt(2 / (input + output))
    return np.random.uniform(-scale, scale, size=(input, output))


def he_init(input: int, output: int):
    scale = np.sqrt(2 / input)
    return np.random.uniform(-scale, scale, size=(input, output))


def random_interval_init(low: float, high: float):
    return lambda input, output: np.random.uniform(low, high, size=(input, output))


initializations = {
    "linear": random_interval_init(-1, 1),
    "sigmoid": xavier_init,
    "tanh": xavier_init,
    "relu": he_init
}
