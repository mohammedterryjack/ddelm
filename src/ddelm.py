"""Deep Diffusion Extreme Learning Machine (DDELM)"""

from numpy import ndarray, argmax, linspace, ones
from numpy.random import seed, uniform
from numpy.linalg import pinv

from src.utils import one_hot_encode, sigmoid


def fit_multi_layered(
    X:ndarray,
    Y:ndarray,
    Ri:ndarray,
    Rh:ndarray,
    Ro:ndarray,
    n_layers:int
) -> list[ndarray]:
    """Fit a multi-layered ELM"""
    h_dim,_ = Rh.shape
    _,o_dim = Y.shape
    noise_levels = linspace(0, 1, num=n_layers+2)[1:-1]
    noise_level = noise_levels[0]

    H1 = sigmoid(X @ Ri)
    Ry1 = uniform(
        low=-noise_levels[0], high=noise_levels[0],
        size=(o_dim,h_dim)
    )
    Y1 = Y @ Ry1
    W1 = pinv(H1) @ Y1
    Y_hat = H1 @ W1
    Ws = [W1]

    for noise_level in noise_levels[1:]:
        Hn = sigmoid(Y_hat @ Rh)
        Ryn = uniform(
            low=-noise_level, high=noise_level,
            size=(o_dim,h_dim)
        )
        Yn = Y @ Ryn
        Wn = pinv(Hn) @ Yn
        Ws.append(Wn)
        Y_hat = Hn @ Wn

    Ho = sigmoid(Y_hat @ Ro)
    Wo = pinv(Ho) @ Y
    Ws.append(Wo)
    return Ws
 

def transform_multi_layered(X:ndarray, Ri:ndarray, Rh:ndarray, Ro:ndarray, Ws:list[ndarray]) -> ndarray:
    H = sigmoid(X@Ri)
    Y_hat = H @ Ws[0]
    for Wn in Ws[1:-1]:
        Hn = sigmoid(Y_hat @ Rh)
        Y_hat = Hn @ Wn
    Ho = sigmoid(Y_hat @ Ro)
    Y_hat = Ho @ Ws[-1]
    return Y_hat

class DDELM:
    def __init__(
        self,
        n_layers:int,
        input_dimension:int,
        output_dimension:int,
        hidden_dimension:int=1000,
    ) -> None:
        seed(42)
        self.Ri = uniform(
            low=-.1, high=.1,
            size=(input_dimension, hidden_dimension)
        )
        self.Rh = uniform(
            low=-.1, high=.1,
            size=(hidden_dimension, hidden_dimension)
        )
        self.Ro = uniform(
            low=-.1, high=.1,
            size=(hidden_dimension, output_dimension)
        )
        self.d_o = output_dimension
        self.n_layers = n_layers 

    def fit(self, X:ndarray, Y:ndarray) -> None:
        Y = one_hot_encode(class_ids=Y,n_classes=self.d_o)
        self.Ws = fit_multi_layered(X=X,Y=Y,Ri=self.Ri,Rh=self.Rh,Ro=self.Ro,n_layers=self.n_layers)

    def predict(self, X:ndarray) -> int:
        Y_hat = transform_multi_layered(X=X,Ri=self.Ri,Rh=self.Rh,Ro=self.Ro,Ws=self.Ws)
        return argmax(Y_hat,axis=1)
