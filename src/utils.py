from enum import Enum

from numpy import (
    ndarray,
    zeros,
    maximum,
    sin,
    arcsin,
    nan_to_num,
    cos,
    arccos,
    tan,
    arctan,
)
from scipy.fft import dct, idct


class Activation(Enum):
    IDENTITY = "identity"
    SIN = "sine"
    COS = "cosine"
    TAN = "tan"
    RELU = "rectified linear unit"
    CT = "cosine transform"


def activation_function(activation: Activation) -> callable:
    return {
        Activation.IDENTITY: lambda x: x,
        Activation.RELU: lambda x: maximum(0, x),
        Activation.SIN: sin,
        Activation.COS: cos,
        Activation.TAN: tan,
        Activation.CT: dct,
    }[activation]


def inverse_activation(activation: Activation) -> callable:
    f = {
        Activation.SIN: arcsin,
        Activation.COS: arccos,
        Activation.TAN: arctan,
        Activation.CT: idct,
    }.get(activation, lambda x: x)
    return lambda x: nan_to_num(f(x), nan=0.0)


def one_hot_encode(class_ids: list[int], n_classes: int) -> ndarray:
    n_samples = len(class_ids)
    vectors = zeros(shape=(n_samples, n_classes), dtype=int)
    vectors[range(n_samples), class_ids] = 1
    return vectors
