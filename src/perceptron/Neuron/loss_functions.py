import numpy as np
from numpy import ndarray


def simple_difference(y: ndarray, y_predicted: ndarray) -> ndarray:
    return y_predicted - y


def binary_cross_entropy(y: ndarray, y_predicted: ndarray) -> ndarray:
    pass


def categorical_cross_entropy(y: ndarray, y_predicted: ndarray) -> ndarray:
    loss = -np.sum(y_predicted * np.log(y))
    return loss / y.shape[0]
