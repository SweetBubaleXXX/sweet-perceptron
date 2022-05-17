from typing import Callable

import numpy as np
from numpy import exp, ndarray

from . import initializations as init_funcs
from . import loss_functions as loss_funcs


def default_initialization(init_func):
    def set_initialization(func):
        func.__initialization__ = init_func
        return func
    return set_initialization


def default_loss(loss_func):
    def set_loss(func):
        func.__loss__ = loss_func
        return func
    return set_loss


@default_initialization(init_funcs.random_interval_init(-1, 1))
@default_loss(loss_funcs.simple_difference)
def linear(x: ndarray, derivative=False) -> ndarray:
    if derivative:
        return 1
    return x


@default_initialization(init_funcs.xavier_init)
@default_loss(loss_funcs.simple_difference)
def sigmoid(x: ndarray, derivative=False) -> ndarray:
    if derivative:
        return exp(-x) / (exp(-x) + 1)**2
    return 1 / (1 + exp(-x))


@default_initialization(init_funcs.xavier_init)
@default_loss(loss_funcs.simple_difference)
def tanh(x: ndarray, derivative=False) -> ndarray:
    if derivative:
        return 4 * exp(2 * x) / (exp(2 * x) + 1)**2
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


class Softmax():
    def __init__(self) -> None:
        self.__name__ = "softmax"
        self.__initialization__ = init_funcs.xavier_init
        self.__loss__ = loss_funcs.categorical_cross_entropy

    def __call__(self, x, derivative=False):
        if derivative:
            return self.input * (x - (x * self.input).sum())
        self.input = x
        return self.calc(x)

    @staticmethod
    def calc(x: ndarray) -> ndarray:
        exps = exp(x - np.max(x))
        return exps / np.sum(exps)


@default_initialization(init_funcs.he_init)
@default_loss(loss_funcs.simple_difference)
def relu(x: ndarray, derivative=False) -> ndarray:
    if derivative:
        return (x > 0).astype(int)
    return np.maximum(0, x)


@default_initialization(init_funcs.he_init)
@default_loss(loss_funcs.simple_difference)
def leaky_relu(x: ndarray, derivative=False) -> ndarray:
    if derivative:
        return np.where(x > 0, 1, 0.01)
    return np.where(x > 0, x, x * 0.01)
