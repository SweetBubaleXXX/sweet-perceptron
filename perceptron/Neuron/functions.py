from turtle import forward
from typing import Callable

from numpy import exp
import numpy as np


def linear(x, derivative=False):
    if derivative:
        return 1
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


def leaky_relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0.01)
    return np.where(x > 0, x, x * 0.01)


def xavier_init(input: int, output: int):
    scale = np.sqrt(2 / (input + output))
    return np.random.uniform(-scale, scale, size=(input, output))


def he_init(input: int, output: int):
    scale = np.sqrt(2 / input)
    return np.random.uniform(-scale, scale, size=(input, output))


def random_interval_init(low: float, high: float) -> Callable[[float, float], np.ndarray]:
    return lambda input, output: np.random.uniform(low, high, size=(input, output))


initializations = {
    "linear": random_interval_init(-1, 1),
    "sigmoid": xavier_init,
    "tanh": xavier_init,
    "relu": he_init,
    "leaky_relu": he_init
}
