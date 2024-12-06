from numpy import ndarray, exp, zeros, maximum, log


def one_hot_encode(class_ids: list[int], n_classes: int) -> ndarray:
    n_samples = len(class_ids)
    vectors = zeros(shape=(n_samples, n_classes), dtype=int)
    vectors[range(n_samples), class_ids] = 1
    return vectors


def sigmoid(x: ndarray) -> ndarray:
    """sigmoid activation function"""
    return 1.0 / (1.0 + exp(-x))

def inverse_sigmoid(x: ndarray) -> ndarray:
    return log(x / (1 - x))
 
def relu(x: ndarray) -> ndarray:
    """rectified linear unit activation function"""
    return maximum(0, x)
