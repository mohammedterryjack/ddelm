"""Deep Diffusion Extreme Learning Machine (DDELM)"""

from numpy import ndarray
from numpy.random import seed
from src.utils import one_hot_encode, sigmoid,  random_noise


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
    noise_levels = linspace(1, 0, num=n_layers+2)[1:-1]

    H1 = sigmoid(X @ Ri)
    Y1 = Y @ random_noise(value=noise_levels[0],shape=(o_dim,h_dim))
    W1 = pinv(H1) @ Y1
    Y_hat = H1 @ W1
    Ws = [W1]

    for noise_level in noise_levels[1:]:
        Hn = sigmoid(Y_hat @ Rh)
        Yn = Y @ random_noise(value=noise_level,shape=(o_dim, h_dim))
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
        self.Ri = random_noise(
            value=.1,shape=(input_dimension, hidden_dimension)
        )
        self.Rh = random_noise(
            value=.1, shape=(hidden_dimension, hidden_dimension)
        )
        self.Ro = random_noise(
            value=.1,shape=(hidden_dimension,output_dimension)
        )
        self.d_o = output_dimension
        self.n_layers = n_layers 

    def fit(self, X:ndarray, Y:ndarray) -> None:
        Y = one_hot_encode(class_ids=Y,n_classes=self.d_o)
        self.Ws = fit_multi_layered(X=X,Y=Y,Ri=self.Ri,Rh=self.Rh,Ro=self.Ro,n_layers=self.n_layers)

    def predict(self, X:ndarray) -> int:
        Y_hat = transform_multi_layered(X=X,Ri=self.Ri,Rh=self.Rh,Ro=self.Ro,Ws=self.Ws)
        return argmax(Y_hat,axis=1)
