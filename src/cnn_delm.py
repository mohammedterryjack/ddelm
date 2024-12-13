"""Convolutional Neural Network (CNN) variation of Deep Extreme Learning Machine (DELM)"""

from numpy import ndarray, argmax, where
from numpy.random import seed, uniform
from numpy.linalg import pinv
from numpy.lib.stride_tricks import as_strided

from src.utils import one_hot_encode, Activation, inverse_activation, activation_function


class CNN:
    def __init__(
        self,
        input_dimension: int,
        kernel_sizes: list[tuple[int,int]],
        hidden_dimensions: list[int],
        output_dimension: int,
        stride:int,
        activation:Activation,
    ) -> None:
        seed(42)
        self.stride=stride
        self.d_i = input_dimension
        self.d_ks = kernel_sizes
        self.d_o = output_dimension
        self.activation = activation_function(activation)
        self.inverse_activation = inverse_activation(activation) 
        d1s,d2s = zip(*kernel_sizes)
        R = uniform(low=-0.1, high=0.1, size=(max(d1s), max(d2s)))
        self.Wks = list(map(lambda ds:R[:ds[0],:ds[1]], self.d_ks))
        X_fake =uniform(low=-0.1,high=0.1,size=(1,self.d_i))
        Yk_final = self.forward_pass_cnn(X=X_fake)
        _,dk_final = Yk_final.shape
        self.dhs = [dk_final] + hidden_dimensions + [output_dimension]
        d1s = self.dhs[:-1]
        d2s = self.dhs[1:]
        R = uniform(low=-0.1, high=0.1, size=(max(d1s), max(d2s)))
        self.Whs = list(map(lambda d1,d2:R[:d1,:d2], d1s, d2s))


    def predict(self, X: ndarray, multilabel_threshold:float|None=None) -> int:
        Y_hat = self.forward_pass_cnn(X=X)
        Y_hat = self.forward_pass_ffnn(X=Y_hat)
        if multilabel_threshold is None:
            return argmax(Y_hat, axis=1)
        return where(Y_hat>multilabel_threshold)

    def forward_pass_cnn(self,X: ndarray) -> ndarray:
        Y_hat = X
        for W in self.Wks:
            window_size,_ = W.shape
            Y_hat_cnn = self.forward_pass_ff_to_cnn_layer(
                ff_layer=Y_hat, 
                window_size=window_size, 
                stride=self.stride
            )
            Y_hat_cnn = self.activation(Y_hat_cnn @ W)
            Y_hat = self.forward_pass_cnn_to_ff_layer(
                cnn_layer=Y_hat_cnn
            )
        return Y_hat

    def forward_pass_ffnn(self, X: ndarray) -> ndarray:
        Y_hat = X
        for W in self.Whs:
            Y_hat = self.activation(Y_hat @ W)
        return Y_hat


    # def fit(self, X: ndarray, Y: ndarray) -> None:
    #     Y = one_hot_encode(class_ids=Y, n_classes=self.d_o)
    #     self.Ws = self.finetune_weights(
    #         X=X,Y=Y,
    #         Ws=self.Ws,
    #         dims=self.dims,activation=self.activation,
    #         inverse_activation=self.inverse_activation
    #     )

    # def finetune_weights(
    #     self,
    #     X: ndarray,
    #     Y: ndarray,
    #     Ws: list[ndarray],
    #     dims: list[int],
    #     activation: callable,
    #     inverse_activation: callable
    # ) -> list[ndarray]:
    #     for i in range(len(Ws)-1):
    #         Y_hat_next = self.backward(
    #             Y=Y,
    #             Ws=Ws[i+1:],
    #             inverse_activation=inverse_activation
    #         )
    #         Y_hat = self.forward(
    #             X=X,
    #             Ws=Ws[:i],
    #             activation=activation
    #         )
    #         Ws[i] = pinv(Y_hat) @ Y_hat_next
    #     return Ws
    
    # @staticmethod
    # def backward_ffnn(Y:ndarray, Ws:list[ndarray], inverse_activation:callable) -> ndarray:
    #     Y_hat = Y
    #     for W in Ws[::-1]:
    #         Y_hat = inverse_activation(Y_hat) @ pinv(W) 
    #     return Y_hat

    # @staticmethod
    # def backward_cnn(Y:ndarray, Ws:list[ndarray], inverse_activation:callable) -> ndarray:
    #     Y_hat = Y
    #     for W in Ws[::-1]:
    #         Y_hat = inverse_activation(Y_hat) @ pinv(W) 
    #     return Y_hat



    @staticmethod
    def forward_pass_ff_to_cnn_layer(ff_layer:ndarray, window_size:int, stride:int) -> ndarray:
        batch_size,layer_width = ff_layer.shape
        n_strides = 1+((layer_width-window_size)//stride) 
        return as_strided(
            ff_layer,
            shape=(batch_size,n_strides,window_size),
            strides=(ff_layer.strides[0],ff_layer.strides[1], ff_layer.strides[1])
        )

    @staticmethod    
    def forward_pass_cnn_to_ff_layer(cnn_layer:ndarray) -> ndarray:
        batch_size, n_strides, window_size = cnn_layer.shape
        return cnn_layer.reshape(batch_size, n_strides * window_size)

    # @staticmethod
    # def backward_pass_ff_to_cnn_layer(cnn_matrix:ndarray) -> ndarray:
    #     batch_size, n_strides, window_size = cnn_matrix.shape
    #     layer_width = n_strides + window_size - 1
    #     reconstructed_layer = zeros((batch_size, layer_width))
    #     overlap_count = zeros((batch_size, layer_width))
    #     for i in range(n_strides):
    #         reconstructed_layer[:, i:i+window_size] += cnn_matrix[:, i, :]
    #         overlap_count[:, i:i+window_size] += 1
    #     return reconstructed_layer / overlap_count

    # @staticmethod
    # def backward_pass_cnn_to_ff_layer(layer: ndarray, window_size: int) -> ndarray:
    #     batch_size, flattened_size = layer.shape        
    #     n_strides = flattened_size // window_size    
    #     return layer.reshape(batch_size, n_strides, window_size)



#TODO: allow for K CNN kernels to be trained (not just 1)
#TODO: extend CNN for 2d image layers
