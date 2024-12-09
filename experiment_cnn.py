from numpy import array, ndarray, zeros
from numpy.lib.stride_tricks import as_strided
from sklearn.datasets import load_breast_cancer, load_digits
from matplotlib.pyplot import subplots, show
from sklearn.metrics import accuracy_score
from numpy.linalg import pinv

from src.utils import Activation
from src.delm import DELM

#TODO: something wrong with the layer to cnn matrix conversion - fix it
#TODO: allow for K CNN kernels to be trained (not just 1)

def layer_to_cnn_matrix(layer:ndarray, window_size:int) -> ndarray:
    batch_size,layer_width = layer.shape
    n_strides = layer_width-window_size+1
    return as_strided(
        layer,
        shape=(batch_size,n_strides,window_size),
        strides=(layer.strides[0],layer.strides[1], layer.strides[1])
    )

def cnn_matrix_to_layer(cnn_matrix:ndarray) -> ndarray:
    batch_size, n_strides, window_size = cnn_matrix.shape
    return cnn_matrix.reshape(batch_size, n_strides * window_size)

def inverse_cnn_matrix_to_layer(layer: ndarray, window_size: int) -> ndarray:
    batch_size, flattened_size = layer.shape        
    n_strides = flattened_size // window_size    
    return layer.reshape(batch_size, n_strides, window_size)

def inverse_layer_to_cnn_matrix(cnn_matrix:ndarray) -> ndarray:
    batch_size, n_strides, window_size = cnn_matrix.shape
    layer_width = n_strides + window_size - 1
    reconstructed_layer = zeros((batch_size, layer_width))
    overlap_count = zeros((batch_size, layer_width))
    for i in range(n_strides):
        reconstructed_layer[:, i:i+window_size] += cnn_matrix[:, i, :]
        overlap_count[:, i:i+window_size] += 1
    return reconstructed_layer / overlap_count


def cnn_backward(Y:ndarray, Ws:list[ndarray], inverse_activation:callable) -> ndarray:
    Y_hat = Y
    print("B")
    for W in Ws[::-1]:
        _,window_size = W.shape
        Y_hat = inverse_cnn_matrix_to_layer(layer=Y_hat,window_size=window_size)
        print("<--",Y_hat.shape)
        Y_hat = inverse_activation(Y_hat) @ pinv(W) 
        Y_hat = inverse_layer_to_cnn_matrix(cnn_matrix=Y_hat)
    print("<--",Y_hat.shape)
    return Y_hat


def cnn_forward(
    X: ndarray,
    Ws: list[ndarray],
    activation: callable,
) -> ndarray:
    Y_hat = X
    print("F")
    for W in Ws:
        window_size,_ = W.shape
        Y_hat = layer_to_cnn_matrix(layer=Y_hat,window_size=window_size)
        print("-->",Y_hat.shape)
        Y_hat = activation(Y_hat @ W)
        Y_hat = cnn_matrix_to_layer(cnn_matrix=Y_hat)
    print("-->",Y_hat.shape)
    return Y_hat


for settings in (
    dict(
        name='breast cancer',
        load_data=load_breast_cancer,
        h_dims=[5,4,3],
        a=Activation.RELU
    ), 
    dict(
        name= 'digits',
        load_data=load_digits,
        h_dims=[5,4,3],
        a=Activation.RELU
    )
):
    data = settings['load_data']()

    X = data.data
    Y = data.target        

    _, d_i = X.shape
    d_o = max(Y) + 1

    cnn = DELM(
        input_dimension=d_i, output_dimension=d_o, 
        hidden_dimensions=settings['h_dims'],
        activation=settings['a']
    )
    cnn.forward = cnn_forward
    cnn.backward = cnn_backward
    cnn.fit(X=X,Y=Y)

    Y_cnn = cnn.predict(X=X)
    accuracy_cnn = accuracy_score(Y, Y_cnn)
    print(accuracy_cnn)

    _, axes = subplots(1, len(cnn.Ws), figsize=(15, 5)) 
    for i, W in enumerate(cnn.Ws):
        axes[i].imshow(W)  
    show()

    if settings['name']=='digits':
        n_samples = 10
        _, axes = subplots(1, n_samples, figsize=(15, 5)) 
        for i in range(n_samples):
            axes[i].imshow(X[i].reshape((8,8)),cmap='gray')
        show()

        y_hat = cnn.forward(X=X,Ws=cnn.Ws,activation=cnn.activation)
        _, axes = subplots(1, 1, figsize=(15, 5)) 
        axes.imshow(y_hat[:n_samples],cmap='gray')
        show()

        classifier_head = DELM(
            input_dimension=y_hat.shape[1], output_dimension=d_o, 
            hidden_dimensions=[300,200,100],
            activation=settings['a']
        )
        classifier_head.fit(X=y_hat,Y=Y)
        Y_hat = classifier_head.predict(X=y_hat)
        accuracy_cnn = accuracy_score(Y, Y_hat)
        print(accuracy_cnn)
