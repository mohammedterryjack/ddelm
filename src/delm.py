"""Deep Extreme Learning Machine (DELM)"""

from numpy import ndarray, argmax, where
from numpy.random import seed, uniform
from numpy.linalg import pinv

from src.utils import one_hot_encode, relu, sigmoid, inverse_sigmoid


class DELM:
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: list[int],
        output_dimension: int,
    ) -> None:
        seed(42)
        self.d_o = output_dimension
        self.dims = [input_dimension] + hidden_dimensions + [output_dimension]
        self.activation = sigmoid #relu
        self.inverse_activation = lambda x:x #inverse_sigmoid

    def fit(self, X: ndarray, Y: ndarray) -> None:
        Y = one_hot_encode(class_ids=Y, n_classes=self.d_o)
        self.Ws = self.learn_weights(X=X,Y=Y,dims=self.dims,activation=self.activation,inverse_activation=self.inverse_activation)

    def predict(self, X: ndarray, multilabel_threshold:float|None=None) -> int:
        Y_hat = self.forward(X=X,Ws=self.Ws,activation=self.activation)
        if multilabel_threshold is None:
            return argmax(Y_hat, axis=1)
        return where(Y_hat>multilabel_threshold)

    def learn_weights(
        self,
        X: ndarray,
        Y: ndarray,
        dims: list[int],
        activation: callable,
        inverse_activation: callable
    ) -> list[ndarray]:
        """Fit a multi-layered ELM"""

        Ws = list(map(lambda d1,d2:uniform(low=-0.1, high=0.1, size=(d1,d2)), dims[:-1],dims[1:]))
        for i in range(len(Ws)-1):
            Y_hat_next = self.backward(
                Y=Y,
                Ws=Ws[i+1:],
                inverse_activation=inverse_activation
            )
            Y_hat = self.forward(
                X=X,
                Ws=Ws[:i],
                activation=activation
            )
            Ws[i] = pinv(Y_hat) @ Y_hat_next
        return Ws
    
    @staticmethod
    def backward(Y:ndarray, Ws:list[ndarray], inverse_activation:callable) -> ndarray:
        Y_hat = Y
        for W in Ws[::-1]:
            Y_hat = inverse_activation(Y_hat) @ pinv(W) 
        return Y_hat


    @staticmethod
    def forward(
        X: ndarray,
        Ws: list[ndarray],
        activation: callable,
    ) -> ndarray:
        Y_hat = X
        for W in Ws:
            Y_hat = activation(Y_hat @ W)
        return Y_hat
