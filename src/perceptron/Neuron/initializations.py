from typing import Callable

import numpy as np
from numpy import ndarray


def xavier_init(input: int, output: int) -> ndarray:
    scale = np.sqrt(2 / (input + output))
    return np.random.uniform(-scale, scale, size=(input, output))


def he_init(input: int, output: int) -> ndarray:
    scale = np.sqrt(2 / input)
    return np.random.uniform(-scale, scale, size=(input, output))


def random_interval_init(low: float, high: float) -> Callable[[float, float], ndarray]:
    return lambda input, output: np.random.uniform(low, high, size=(input, output))
