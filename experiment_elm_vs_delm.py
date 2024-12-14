from sklearn.datasets import load_breast_cancer
from matplotlib.pyplot import subplots, show, scatter, legend, xlabel, ylabel
from sklearn.metrics import accuracy_score

from src.utils import Activation
from src.elm import ELM
from src.delm import DELM

data = load_breast_cancer()

X = data.data
Y = data.target

_, d_i = X.shape
d_o = max(Y) + 1


elm = ELM(
    input_dimension=d_i,
    output_dimension=d_o,
    hidden_dimension=50,
    activation=Activation.SIN,
)
delm = DELM(
    input_dimension=d_i,
    output_dimension=d_o,
    hidden_dimensions=[50],
    activation=Activation.SIN,
)

elm.fit(X=X, y=Y)
delm.fit(X=X, y=Y)

Y_elm = elm.predict(X=X)
Y_delm = delm.predict(X=X)

accuracy_elm = accuracy_score(Y, Y_elm)
accuracy_delm = accuracy_score(Y, Y_delm)

print(f"ELM:{accuracy_elm}\nDELM:{accuracy_delm}")

_, axes = subplots(2, 2, figsize=(15, 5))
axes[0, 0].set_ylabel("ELM")
axes[1, 0].set_ylabel("DELM")
for i, (W_elm, W_delm) in enumerate(zip([elm.R, elm.W], delm.Ws)):
    axes[0, i].imshow(W_elm)
    axes[1, i].imshow(W_delm)
    axes[1, i].set_xlabel(f"W {i}")
show()


accuracies_elm = []
accuracies_delm = []
h_dims = list(range(10, 100, 10))
for h_dim in h_dims:
    elm = ELM(
        input_dimension=d_i,
        output_dimension=d_o,
        hidden_dimension=h_dim,
        activation=Activation.SIN,
    )
    delm = DELM(
        input_dimension=d_i,
        output_dimension=d_o,
        hidden_dimensions=[h_dim],
        activation=Activation.SIN,
    )

    elm.fit(X=X, y=Y)
    delm.fit(X=X, y=Y)

    Y_elm = elm.predict(X=X)
    Y_delm = delm.predict(X=X)

    accuracies_elm.append(accuracy_score(Y, Y_elm))
    accuracies_delm.append(accuracy_score(Y, Y_delm))

scatter(h_dims, accuracies_elm, label="ELM")
scatter(h_dims, accuracies_delm, label="DELM")
ylabel("accuracy")
xlabel("hidden layer dimension")
legend()
show()
