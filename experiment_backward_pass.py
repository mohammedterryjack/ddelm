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

for a in Activation:
    if a != Activation.RELU:
        continue
    delm = DELM(
        input_dimension=d_i, output_dimension=d_o, hidden_dimensions=[100], activation=a
    )
    delm.fit(X=X, y=Y)

    y_hat = delm.forward(X=X_test, Ws=delm.Ws, activation=delm.activation)
    _, axes = subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(y_test, cmap="pink")
    axes[1].imshow(y_hat, cmap="pink")
    axes[2].imshow(y_test - y_hat, cmap="pink")
    axes[0].set_title("y (target)")
    axes[1].set_title("y (predicted)")
    axes[2].set_title("dy (error)")
    show()

    X_hat = delm.backward(
        Y=y_hat, Ws=delm.Ws, inverse_activation=delm.inverse_activation
    )
    _, axes = subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(X_test, cmap="pink")
    axes[1].imshow(X_hat, cmap="pink")
    axes[2].imshow(X_test - X_hat, cmap="pink")
    axes[0].set_title("X (given)")
    axes[1].set_title("X (from y_hat)")
    axes[2].set_title("dX (error)")
    show()

    X_hat = delm.backward(
        Y=y_test, Ws=delm.Ws, inverse_activation=delm.inverse_activation
    )
    _, axes = subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(X_test, cmap="pink")
    axes[1].imshow(X_hat, cmap="pink")
    axes[2].imshow(X_test - X_hat, cmap="pink")
    axes[0].set_title("X (given)")
    axes[1].set_title("X (from y)")
    axes[2].set_title("dX (error)")
    show()
