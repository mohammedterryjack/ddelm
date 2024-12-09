from numpy import array, ndarray, zeros
from numpy.lib.stride_tricks import as_strided
from sklearn.datasets import load_breast_cancer, load_digits
from matplotlib.pyplot import subplots, show
from sklearn.metrics import accuracy_score
from numpy.linalg import pinv

from src.utils import Activation
from src.delm import DELM

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
    print(f"B: {[W.shape for W in Ws]}<--")
    for i,W in enumerate(Ws[::-1]):
        if i: #final layer is dense - not CNN
            print("<--",Y_hat.shape)
            _,window_size = W.shape
            Y_hat = inverse_cnn_matrix_to_layer(layer=Y_hat, window_size=window_size)
            print(".1.",Y_hat.shape)
        Y_hat = inverse_activation(Y_hat) @ pinv(W) 
        print("<-2-",Y_hat.shape)
        if i:            
            Y_hat = inverse_layer_to_cnn_matrix(cnn_matrix=Y_hat)
            print(".3.",Y_hat.shape)
    print("<--",Y_hat.shape)
    return Y_hat


def cnn_forward(
    X: ndarray,
    Ws: list[ndarray],
    activation: callable,
) -> ndarray:
    Y_hat = X
    print(f"F: -->{[W.shape for W in Ws]}")
    for W in Ws:
        print("-->",Y_hat.shape)
        window_size,_ = W.shape
        Y_hat = layer_to_cnn_matrix(layer=Y_hat,window_size=window_size)
        print(".1.",Y_hat.shape)
        Y_hat = activation(Y_hat @ W)
        print("-2->",Y_hat.shape)
        Y_hat = cnn_matrix_to_layer(cnn_matrix=Y_hat)
        print(".3.",Y_hat.shape)
    print("-->",Y_hat.shape)
    return Y_hat


for settings in (
    dict(
        name='breast cancer',
        load_data=load_breast_cancer,
        h_dims=[9,10,100], #Final Layer must be larger than d_o and dense (not cnn).  Penultimate layer must be multiple of final layer (e.g. 99).  First layer must be same as d_i
        a=Activation.RELU
    ), 
    dict(
        name= 'digits',
        load_data=load_digits,
        h_dims=[9,10,100], #[(64, 18), (18, 100), (100, 10)] = [( d_i, 2*d_h1 ), (2*d_h1, d_h3), (d_h3,d_o)]
        a=Activation.RELU
    )
):
    data = settings['load_data']()

    X = data.data
    Y = data.target        

    _, d_i = X.shape
    d_o = max(Y) + 1
    print(d_i,d_o)

    cnn = DELM(
        input_dimension=d_i, output_dimension=d_o, 
        hidden_dimensions=settings['h_dims'],
        activation=settings['a']
    )
    cnn.forward = cnn_forward
    cnn.backward = cnn_backward
    del cnn.Ws[-2]
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

#TODO: allow for K CNN kernels to be trained (not just 1)
#TODO: extend CNN for 2d image layers
