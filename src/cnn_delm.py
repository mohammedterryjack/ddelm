"""Convolutional Neural Network (CNN) variation of Deep Extreme Learning Machine (DELM)"""

from numpy import ndarray, argmax, where, zeros, divide
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
        Yk_final = self.forward_pass_cnn(X=X_fake, Ws=self.Wks, activation=self.activation, stride=self.stride)
        _,dk_final = Yk_final.shape
        self.dhs = [dk_final] + hidden_dimensions + [output_dimension]
        d1s = self.dhs[:-1]
        d2s = self.dhs[1:]
        R = uniform(low=-0.1, high=0.1, size=(max(d1s), max(d2s)))
        self.Whs = list(map(lambda d1,d2:R[:d1,:d2], d1s, d2s))


    def predict(self, X: ndarray, multilabel_threshold:float|None=None) -> int:
        Y_hat = self.forward_pass_cnn(X=X, Ws=self.Wks, stride=self.stride, activation=self.activation)
        Y_hat = self.forward_pass_ffnn(X=Y_hat, Ws=self.Whs, activation=self.activation)
        if multilabel_threshold is None:
            return argmax(Y_hat, axis=1)
        return where(Y_hat>multilabel_threshold)

    def fit(self, X: ndarray, y: ndarray) -> None:
        forward_pass = lambda layer:self.forward_pass_cnn(
            X=X,
            Ws=self.Wks[:layer],
            activation=self.activation,
            stride=self.stride
        ) if layer < len(self.Wks) else self.forward_pass_ffnn(
            X=self.forward_pass_cnn(
                X=X,
                Ws=self.Wks,
                activation=self.activation,
                stride=self.stride
            ),
            Ws=self.Whs[:layer-len(self.Wks)],
            activation=self.activation,
        )
        backward_pass = lambda layer:self.backward_pass_cnn(
            Y=self.backward_pass_ffnn(
                Y=Y,
                Ws=self.Whs,
                inverse_activation=self.inverse_activation
            ),
            Ws=self.Wks[layer+1:],
            inverse_activation=self.inverse_activation,
            stride=self.stride
        ) if layer < len(self.Wks) else self.backward_pass_ffnn(
            Y=Y,
            Ws=self.Whs[layer+1-len(self.Wks):],
            inverse_activation=self.inverse_activation,
        )
        Y = one_hot_encode(class_ids=y, n_classes=self.d_o)
        Ws_finetuned = [
            pinv(forward_pass(layer=layer_i)) @ backward_pass(layer=layer_i)
            for layer_i in range(len(self.Wks)+len(self.Whs)-1)
        ]  
        self.Wks = Ws_finetuned[:len(self.Wks)]
        self.Whs = Ws_finetuned[len(self.Wks):]        

    @staticmethod
    def forward_pass_cnn(X: ndarray, Ws:list[ndarray], stride:int, activation:callable) -> ndarray:
        def forward_pass_ff_to_cnn_layer(ff_layer:ndarray, window_size:int, stride:int) -> ndarray:
            batch_size,layer_width = ff_layer.shape
            n_strides = 1+((layer_width-window_size)//stride) 
            return as_strided(
                ff_layer,
                shape=(batch_size,n_strides,window_size),
                strides=(ff_layer.strides[0],ff_layer.strides[1], ff_layer.strides[1])
            )

        def forward_pass_cnn_to_ff_layer(cnn_layer:ndarray) -> ndarray:
            batch_size, n_strides, window_size = cnn_layer.shape
            return cnn_layer.reshape(batch_size, n_strides * window_size)


        Y_hat = X
        for W in Ws:
            window_size,_ = W.shape
            Y_hat_cnn = forward_pass_ff_to_cnn_layer(
                ff_layer=Y_hat, 
                window_size=window_size, 
                stride=stride
            )
            Y_hat_cnn = activation(Y_hat_cnn @ W)
            Y_hat = forward_pass_cnn_to_ff_layer(
                cnn_layer=Y_hat_cnn
            )
        return Y_hat

    @staticmethod
    def forward_pass_ffnn(X: ndarray, Ws:list[ndarray], activation:callable) -> ndarray:
        Y_hat = X
        for W in Ws:
            Y_hat = activation(Y_hat @ W)
        return Y_hat

    @staticmethod
    def backward_pass_cnn(Y: ndarray, Ws:list[ndarray], stride:int, inverse_activation:callable) -> ndarray:
        def backward_pass_cnn_to_ff_layer(ff_layer: ndarray, window_size: int) -> ndarray:
            batch_size, layer_size = ff_layer.shape        
            return ff_layer.reshape(batch_size, layer_size // window_size, window_size)

        def backward_pass_ff_to_cnn_layer(cnn_layer: ndarray, stride: int) -> ndarray:
            batch_size, n_strides, window_size = cnn_layer.shape
            layer_width = (n_strides - 1) * stride + window_size
            ff_layer = zeros((batch_size, layer_width))
            counts = zeros((batch_size, layer_width))  
            
            for i in range(n_strides):
                start = i * stride
                end = start + window_size
                ff_layer[:, start:end] += cnn_layer[:, i, :]
                counts[:, start:end] += 1
            
            ff_layer = divide(ff_layer, counts, where=(counts != 0))        
            return ff_layer

        Y_hat = Y
        for W in Ws[::-1]:
            _,window_size = W.shape
            Y_hat_cnn = backward_pass_cnn_to_ff_layer(ff_layer=Y_hat, window_size=window_size)
            Y_hat_cnn = inverse_activation(Y_hat_cnn) @ pinv(W) 
            Y_hat = backward_pass_ff_to_cnn_layer(cnn_layer=Y_hat_cnn, stride=stride)
        return Y_hat

    @staticmethod
    def backward_pass_ffnn(Y: ndarray, Ws:list[ndarray], inverse_activation:callable) -> ndarray:
        Y_hat = Y
        for W in Ws[::-1]:
            Y_hat = inverse_activation(Y_hat) @ pinv(W)
        return Y_hat




