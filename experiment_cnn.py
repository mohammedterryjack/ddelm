from sklearn.datasets import load_digits

from src.utils import Activation
from src.cnn_delm import CNN


data = load_digits()
X = data.data
Y = data.target        
_, d_i = X.shape
d_o = max(Y) + 1

cnn = CNN(
    input_dimension=d_i, output_dimension=d_o, 
    kernel_sizes=[(3,4),(7,6)],
    hidden_dimensions=[100],
    stride=1,
    activation=Activation.RELU
)
