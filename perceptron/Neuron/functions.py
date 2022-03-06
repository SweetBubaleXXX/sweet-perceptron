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
