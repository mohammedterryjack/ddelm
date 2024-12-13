from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

from src.utils import Activation
from src.cnn_delm import CNN


data = load_digits()
X = data.data
Y = data.target        
_, d_i = X.shape
d_o = max(Y) + 1

cnn = CNN(
    input_dimension=d_i, output_dimension=d_o, 
    kernel_sizes=[(5,4),(7,6)],
    hidden_dimensions=[100,30],
    stride=1,
    activation=Activation.RELU
)
cnn.fit(X=X,y=Y)
Y_cnn = cnn.predict(X=X)

accuracy_cnn = accuracy_score(Y, Y_cnn)
print(f"CNN: {accuracy_cnn}")