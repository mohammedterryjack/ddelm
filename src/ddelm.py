"""Deep Diffusion Extreme Learning Machine (DDELM)"""

from numpy import ndarray, argmax, linspace, ones
from numpy.random import seed, uniform
from numpy.linalg import pinv

from src.utils import one_hot_encode, sigmoid


def fit_multi_layered(
    X:ndarray,
    Y:ndarray,
    Rs:list[ndarray],
) -> list[ndarray]:
    """Fit a multi-layered ELM"""
    _,o_dim = Y.shape
    noise_levels = linspace(0, 1, num=len(Rs)+2)[1:-1]
    noise_level = noise_levels[0]

    Ri = Rs[0]
    Rhs = Rs[1:-1]
    Ro = Rs[-1]

    h_dim_next,_ = Rhs[0].shape

    H1 = sigmoid(X @ Ri)
    Ry1 = uniform(
        low=-noise_levels[0], high=noise_levels[0],
        size=(o_dim,h_dim_next)
    )
    Y1 = Y @ Ry1
    W1 = pinv(H1) @ Y1
    Y_hat = H1 @ W1
    Ws = [W1]

    for noise_level,layer in zip(noise_levels[1:],range(len(Rhs)-1)):
        h_dim_next,_ = Rhs[layer+1].shape
        Hn = sigmoid(Y_hat @ Rhs[layer])
        Ryn = uniform(
            low=-noise_level, high=noise_level,
            size=(o_dim,h_dim_next)
        )
        Yn = Y @ Ryn
        Wn = pinv(Hn) @ Yn
        Ws.append(Wn)
        Y_hat = Hn @ Wn

    Ho = sigmoid(Y_hat @ Ro)
    Wo = pinv(Ho) @ Y
    Ws.append(Wo)
    return Ws
 

def transform_multi_layered(X:ndarray,Rs:list[ndarray], Ws:list[ndarray]) -> ndarray:
    Ri = Rs[0]
    Rhs = Rs[1:-1]
    Ro = Rs[-1]

    H = sigmoid(X@Ri)
    Y_hat = H @ Ws[0]
    for Wn,Rh in zip(Ws[1:-1],Rhs):
        Hn = sigmoid(Y_hat @ Rh)
        Y_hat = Hn @ Wn
    Ho = sigmoid(Y_hat @ Ro)
    Y_hat = Ho @ Ws[-1]
    return Y_hat

class DDELM:
    def __init__(
        self,
        input_dimension:int,
        output_dimension:int,
        hidden_dimensions:list[int],
    ) -> None:
        seed(42)
        self.Rs = [
            uniform(
                low=-.1, high=.1,
                size=(input_dimension, input_dimension)
            )
        ] + [
            uniform(
                low=-.1, high=.1,
                size=(hidden_dimensions[i],hidden_dimensions[i])
            ) for i in range(len(hidden_dimensions))
        ] + [ 
            uniform(
                low=-.1, high=.1,
                size=(hidden_dimensions[-1], hidden_dimensions[-1])
            )
        ]
        self.d_o = output_dimension

    def fit(self, X:ndarray, Y:ndarray) -> None:
        Y = one_hot_encode(class_ids=Y,n_classes=self.d_o)
        self.Ws = fit_multi_layered(X=X,Y=Y,Rs=self.Rs)

    def predict(self, X:ndarray) -> int:
        Y_hat = transform_multi_layered(X=X,Rs=self.Rs,Ws=self.Ws)
        return argmax(Y_hat,axis=1)
