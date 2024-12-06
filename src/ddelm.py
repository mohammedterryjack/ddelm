"""Deep Diffusion Extreme Learning Machine (DDELM)"""

from numpy import ndarray, argmax
from numpy.random import seed, uniform
from numpy.linalg import pinv

from src.utils import one_hot_encode, relu, sigmoid


class DDELM:
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: list[int],
        output_dimension: int,
        activation_function: callable = sigmoid#relu
    ) -> None:
        seed(42)
        largest_dimension = max(
            [input_dimension] + hidden_dimensions + [output_dimension]
        )
        self.R = uniform(
            low=-0.1, high=0.1, size=(largest_dimension, largest_dimension)
        )
        self.d_i = input_dimension
        self.d_hs = hidden_dimensions
        self.d_o = output_dimension
        self.a = activation_function

    def fit(self, X: ndarray, Y: ndarray) -> None:
        Y = one_hot_encode(class_ids=Y, n_classes=self.d_o)
        self.Ws = self.fit_multi_layered(
            X=X,
            Y=Y,
            R=self.R,
            layer_dimensions=[self.d_i] + self.d_hs,
            activation=self.a,
        )


    def predict(self, X: ndarray) -> int:
        Y_hat = self.transform_multi_layered(
            X=X,
            R=self.R,
            Ws=self.Ws,
            layer_dimensions=[self.d_i] + self.d_hs,
            activation=self.a,
        )
        return argmax(Y_hat, axis=1)

    @staticmethod
    def fit_multi_layered(
        X: ndarray,
        Y: ndarray,
        R: ndarray,
        layer_dimensions: list[int],
        activation: callable,
    ) -> list[ndarray]:
        """Fit a multi-layered ELM"""
        _, o_dim = Y.shape

        Ws = []
        Y_hat = X
        for layer in range(len(layer_dimensions) - 1):
            h_dim_next = layer_dimensions[layer + 1]
            Rh = R[: layer_dimensions[layer], : layer_dimensions[layer]]
            H = activation(Y_hat @ Rh)
            Ry = R[:o_dim, :h_dim_next]
            Yr = Y @ Ry
            W = pinv(H) @ Yr
            Ws.append(W)
            Y_hat = H @ W

        Ro = R[: layer_dimensions[-1], : layer_dimensions[-1]]
        Ho = activation(Y_hat @ Ro)
        Wo = pinv(Ho) @ Y
        Ws.append(Wo)
        return Ws

    @staticmethod
    def transform_multi_layered(
        X: ndarray,
        Ws: list[ndarray],
        R: ndarray,
        activation: callable,
        layer_dimensions: list[int],
    ) -> ndarray:
        Y_hat = X
        for W, h_dim in zip(Ws, layer_dimensions):
            H = activation(Y_hat @ R[:h_dim, :h_dim])
            Y_hat = H @ W
        return Y_hat
