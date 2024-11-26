
from numpy import ndarray, exp, linspace, argmax, zeros
from numpy.random import uniform, seed
from numpy.linalg import pinv

 
def one_hot_encode(class_ids:list[int], n_classes:int) -> ndarray:
    n_samples=len(class_ids)
    vectors = zeros(shape=(n_samples,n_classes),dtype=int)
    vectors[range(n_samples),class_ids]=1
    return vectors

 
def sigmoid(x: ndarray) -> ndarray:
    """activation function"""
    return 1. / (1. + exp(-x))
 

def random_noise(value:float,shape:tuple[int,int]) -> ndarray:
    return uniform(
        low=-value, high=value,
        size=shape
    )
