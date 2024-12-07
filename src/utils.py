from enum import Enum

from numpy import ndarray, zeros, maximum, sin,arcsin,nan_to_num

class Activation(Enum):
    SINE="sine"
    RELU="relu" 

def activation_function(activation:Activation) -> callable:
    return {
        Activation.SINE:sin,
        Activation.RELU:lambda x:maximum(0, x)
    }[activation]

def inverse_activation(activation:Activation) -> callable:
    return {
        Activation.SINE:lambda x:nan_to_num(arcsin(x),nan=0.0),
    }.get(activation, lambda x:x)

def one_hot_encode(class_ids: list[int], n_classes: int) -> ndarray:
    n_samples = len(class_ids)
    vectors = zeros(shape=(n_samples, n_classes), dtype=int)
    vectors[range(n_samples), class_ids] = 1
    return vectors
