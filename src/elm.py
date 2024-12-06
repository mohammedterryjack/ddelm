"""Extreme Learning Machine (ELM)"""

from numpy import ndarray, argmax
from numpy.random import seed, uniform
from numpy.linalg import pinv

from src.utils import one_hot_encode, relu, sigmoid


class ELM:
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_dimension: int = 1000,
        activation_function: callable = sigmoid #relu
    ) -> None:
        seed(42)

        self.R = uniform(low=-0.1, high=1.0, size=(input_dimension, hidden_dimension))
        self.d_o = output_dimension
        self.a = activation_function

    def fit(self, X: ndarray, Y: ndarray) -> None:
        Y = one_hot_encode(class_ids=Y, n_classes=self.d_o)
        self.W = self.fit_single_layer(X=X, Y=Y, R=self.R, activation=self.a)

    def predict(self, X: ndarray) -> int:
        Y_hat = self.transform_single_layer(X=X, R=self.R, W=self.W, activation=self.a)
        return argmax(Y_hat, axis=1)

    @staticmethod
    def fit_single_layer(
        X: ndarray, Y: ndarray, R: ndarray, activation: callable
    ) -> ndarray:
        """Fit a single Layer ELM using Pseudo-Inverse"""
        H = activation(X @ R)
        return pinv(H) @ Y

    @staticmethod
    def transform_single_layer(
        X: ndarray, R: ndarray, W: ndarray, activation: callable
    ) -> ndarray:
        H = activation(X @ R)
        return H @ W
