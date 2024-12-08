from sklearn.datasets import load_breast_cancer
from matplotlib.pyplot import imshow, show,title

from src.utils import Activation, one_hot_encode
from src.delm import DELM

data = load_breast_cancer()

X = data.data
Y = data.target

_, d_i = X.shape
d_o = max(Y) + 1

for a in Activation:
    delm = DELM(
        input_dimension=d_i, output_dimension=d_o, 
        hidden_dimensions=[100,100],
        activation=Activation.SIN
    )

    delm.fit(X=X, Y=Y)

    y = one_hot_encode(class_ids=Y, n_classes=d_o)
    X_hat = delm.backward(
        Y=y,
        Ws=delm.Ws,
        inverse_activation=delm.inverse_activation
    )
    deltaX = X-X_hat

    title(a.value)
    imshow(deltaX > 1e-3)
    show()
