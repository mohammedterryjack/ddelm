from matplotlib.pyplot import subplots, show
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.neural_network import MLPClassifier

from src.utils import Activation
from src.elm import ELM
from src.ffnn import FFNN
from src.cnn import CNN


for dataset in (
    dict(
        name="breast cancer",
        loader=load_breast_cancer,
        elm_dimensions=100,
        ffnn_dimensions=[40, 30, 20, 10],
    ),
    dict(
        name="digits",
        loader=load_digits,
        elm_dimensions=1000,
        ffnn_dimensions=[400, 300, 200, 100],
    ),
):
    continue

    # if a in (Activation.IDENTITY, Activation.RELU):
    #         dnn = MLPClassifier(
    #             hidden_layer_sizes=dataset["ffnn_dimensions"], activation=a.name.lower()
    #         )
    #         dnn.fit(X=X_train, y=Y_train)
    #         for i, W in enumerate(dnn.coefs_):
    #             ax = dnn_axes[j, i]
    #             ax.imshow(W)
    #             if i == 0:
    #                 ax.set_ylabel(a.name)
    #             if j == len(Activation) - 1:
    #                 ax.set_xlabel(f"W {i}")
    #         Y_dnn = dnn.predict(X=X_test)
    #         accuracy_dnn = accuracy_score(Y_test, Y_dnn)
    #     else:
    #         accuracy_dnn = -1.0