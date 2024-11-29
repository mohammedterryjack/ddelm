"""Deep Diffusion Extreme Learning Machine (DDELM)"""

from numpy import ndarray, argmax, linspace
from numpy.random import seed, uniform
from numpy.linalg import pinv

from src.utils import one_hot_encode, sigmoid, relu


def fit_multi_layered(
    X:ndarray,
    Y:ndarray,
    Rs:list[ndarray],
    activation:callable
) -> list[ndarray]:
    """Fit a multi-layered ELM"""
    _,o_dim = Y.shape
    noise_levels = linspace(0, 1, num=len(Rs)+2)[1:-1]

    Ws = []
    Y_hat = X
    for noise_level,layer in zip(noise_levels,range(len(Rs)-1)):
        h_dim_next,_ = Rs[layer+1].shape
        H = activation(Y_hat @ Rs[layer])
        Ry = uniform(
            low=-noise_level, high=noise_level,
            size=(o_dim,h_dim_next)
        )
        Yr = Y @ Ry
        W = pinv(H) @ Yr
        Ws.append(W)
        Y_hat = H @ W

    Ho = activation(Y_hat @ Rs[-1])
    Wo = pinv(Ho) @ Y
    Ws.append(Wo)
    return Ws
 

def transform_multi_layered(X:ndarray,Rs:list[ndarray], Ws:list[ndarray], activation:callable) -> ndarray:
    Y_hat = X
    for W,R in zip(Ws,Rs):
       H = activation(Y_hat @ R)
       Y_hat = H @ W
    return Y_hat

class DDELM:
    def __init__(
        self,
        input_dimension:int,
        hidden_dimensions:list[int],
        output_dimension:int,
        activation_function:callable=relu #sigmoid
    ) -> None:
        seed(42)
        self.Rs = [
            uniform(
                low=-.1, high=.1,
                size=(hidden_dimension,hidden_dimension)
            ) for hidden_dimension in [input_dimension] + hidden_dimensions
        ] 
        self.d_o = output_dimension
        self.a = activation_function

    def fit(self, X:ndarray, Y:ndarray) -> None:
        Y = one_hot_encode(class_ids=Y,n_classes=self.d_o)
        self.Ws = fit_multi_layered(X=X,Y=Y,Rs=self.Rs, activation=self.a)

    def predict(self, X:ndarray) -> int:
        Y_hat = transform_multi_layered(X=X,Rs=self.Rs,Ws=self.Ws, activation=self.a)
        return argmax(Y_hat,axis=1)
