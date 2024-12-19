from enum import Enum

from numpy import (
    ndarray,
    zeros,
    maximum,
    nan_to_num,
    tan,
    arctan,
)


class Activation(Enum):
    IDENTITY = "identity"
    RELU = "rectified linear unit"
    TAN = "tan"


def activation_function(activation: Activation) -> callable:
    return {
        Activation.IDENTITY: lambda x: x,
        Activation.RELU: lambda x: maximum(0, x),
        Activation.TAN: tan,
    }[activation]


def inverse_activation(activation: Activation) -> callable:
    f = {
        Activation.TAN: arctan,
    }.get(activation, lambda x: x)
    return lambda x: nan_to_num(f(x), nan=0.0)


def one_hot_encode(class_ids: list[int], n_classes: int) -> ndarray:
    n_samples = len(class_ids)
    vectors = zeros(shape=(n_samples, n_classes), dtype=int)
    vectors[range(n_samples), class_ids] = 1
    return vectors
