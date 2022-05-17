from typing import Callable

from numpy import exp
import numpy as np

from . import initializations as init_funcs


def default_initialization(init_func: Callable) -> Callable:
    def set_initialization(func: Callable) -> Callable:
        func.__initialization__ = init_func
        return func
    return set_initialization


@default_initialization(init_funcs.random_interval_init(-1, 1))
def linear(x, derivative=False):
    if derivative:
        return 1
    return x


@default_initialization(init_funcs.xavier_init)
def sigmoid(x, derivative=False):
    if derivative:
        return exp(-x) / (exp(-x) + 1)**2
    return 1 / (1 + exp(-x))


@default_initialization(init_funcs.xavier_init)
def tanh(x, derivative=False):
    if derivative:
        return 4 * exp(2 * x) / (exp(2 * x) + 1)**2
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


class Softmax():
    def __init__(self) -> None:
        self.__name__ = "softmax"
        self.__initialization__ = init_funcs.xavier_init

    def __call__(self, x, derivative=False):
        if derivative:
            return self.input * (x - (x * self.input).sum())
        self.input = x
        return self.calc(x)

    @staticmethod
    def calc(x):
        exps = exp(x - np.max(x))
        return exps / np.sum(exps)


@default_initialization(init_funcs.he_init)
def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(int)
    return np.maximum(0, x)


@default_initialization(init_funcs.he_init)
def leaky_relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0.01)
    return np.where(x > 0, x, x * 0.01)
