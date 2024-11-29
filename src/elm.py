"""Extreme Learning Machine (ELM)"""

from numpy import ndarray, argmax
from numpy.random import seed, uniform
from numpy.linalg import pinv

from src.utils import one_hot_encode, sigmoid, relu

def fit_single_layer(X:ndarray,Y:ndarray,R:ndarray, activation:callable) -> ndarray:
    """Fit a single Layer ELM using Pseudo-Inverse"""
    H = activation(X @ R)
    return pinv(H) @ Y
 

def transform_single_layer(X:ndarray, R:ndarray, W:ndarray, activation:callable) -> ndarray:
    H = activation(X @ R)
    return H @ W

 
class ELM:
    def __init__(
        self,
        input_dimension:int,
        output_dimension:int,
        hidden_dimension:int=1000,
        activation_function:callable=relu #sigmoid
    ) -> None:
        seed(42)

        self.R = uniform(
            low=-.1, high=1.,
            size=(input_dimension, hidden_dimension)
        )        
        self.d_o = output_dimension 
        self.a = activation_function

    def fit(self, X:ndarray, Y:ndarray) -> None:
        Y = one_hot_encode(class_ids=Y,n_classes=self.d_o)
        self.W = fit_single_layer(X=X,Y=Y,R=self.R,activation=self.a)

    def predict(self, X:ndarray) -> int:
        Y_hat = transform_single_layer(X=X,R=self.R,W=self.W,activation=self.a)
        return argmax(Y_hat,axis=1)

 


# print(f"Accuracy: \n\tELM:{accuracy_elm * 100:.2f}%\n\tDDELM:{accuracy_ddelm * 100:.2f}%")

# #ELM:92.27%

# #DDELM:98.42%
