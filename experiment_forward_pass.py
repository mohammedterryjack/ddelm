from sklearn.datasets import load_breast_cancer
from matplotlib.pyplot import subplots, show

from src.utils import Activation, one_hot_encode
from src.delm import DELM

data = load_breast_cancer()

X = data.data
Y = data.target

_, d_i = X.shape
d_o = max(Y) + 1

y = one_hot_encode(class_ids=Y, n_classes=d_o)

X_test = X[:100]
y_test = y[:100]

delm = DELM(
    input_dimension=d_i, output_dimension=d_o, 
    hidden_dimensions=[10,20,30],
    activation=Activation.RELU
)
delm.fit(X=X, Y=Y)

_, axes = subplots(1, 2*len(delm.Ws)-1, figsize=(15, 5)) 

for i in range(len(delm.Ws)-1):
    j = 2*i
    Y_hat = delm.forward(
        X=X,
        Ws=delm.Ws[:i],
        activation=delm.activation
    )
    axes[j].imshow(Y_hat,cmap='pink')
    axes[j+1].imshow(delm.Ws[i])
axes[2*len(delm.Ws)-2].imshow(Y_hat,cmap='pink')
show()