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
    data = dataset["loader"]()
    X_train, X_test, Y_train, Y_test = train_test_split(
        data.data, data.target, test_size=0.1, random_state=42
    )

    _, elm_axes = subplots(len(Activation), 2, figsize=(15, 5))
    _, ffnn_axes = subplots(
        len(Activation), len(dataset["ffnn_dimensions"]) + 1, figsize=(15, 5)
    )
    _, dnn_axes = subplots(
        len(Activation), len(dataset["ffnn_dimensions"]) + 1, figsize=(15, 5)
    )
    for j, a in enumerate(Activation):
        _, d_i = data.data.shape
        d_o = max(data.target) + 1

        elm = ELM(
            input_dimension=d_i,
            output_dimension=d_o,
            hidden_dimension=dataset["elm_dimensions"],
            activation=a,
        )
        elm.fit(X=X_train, y=Y_train)
        for i, W in enumerate([elm.R, elm.W]):
            ax = elm_axes[j, i]
            ax.imshow(W)
            if i == 0:
                ax.set_ylabel(a.name)
            if j == len(Activation) - 1:
                ax.set_xlabel(f"W {i}")
        Y_elm = elm.predict(X=X_test)
        accuracy_elm = accuracy_score(Y_test, Y_elm)

        ffnn = FFNN(
            input_dimension=d_i,
            output_dimension=d_o,
            hidden_dimensions=dataset["ffnn_dimensions"],
            activation=a,
        )
        ffnn.fit(X=X_train, y=Y_train)
        for i, W in enumerate(ffnn.Ws):
            ax = ffnn_axes[j, i]
            ax.imshow(W)
            if i == 0:
                ax.set_ylabel(a.name)
            if j == len(Activation) - 1:
                ax.set_xlabel(f"W {i}")
        Y_ffnn = ffnn.predict(X=X_test)
        accuracy_ffnn = accuracy_score(Y_test, Y_ffnn)

        cnn = CNN(
            input_dimension=d_i,
            output_dimension=d_o,
            hidden_dimensions=dataset["ffnn_dimensions"],
            kernel_sizes=[(5, 4), (7, 6)],
            stride=1,
            activation=a,
        )
        cnn.fit(X=X_train, y=Y_train)
        Y_cnn = cnn.predict(X=X_test)
        accuracy_cnn = accuracy_score(Y_test, Y_cnn)

        if a in (Activation.IDENTITY, Activation.RELU):
            dnn = MLPClassifier(
                hidden_layer_sizes=dataset["ffnn_dimensions"], activation=a.name.lower()
            )
            dnn.fit(X=X_train, y=Y_train)
            for i, W in enumerate(dnn.coefs_):
                ax = dnn_axes[j, i]
                ax.imshow(W)
                if i == 0:
                    ax.set_ylabel(a.name)
                if j == len(Activation) - 1:
                    ax.set_xlabel(f"W {i}")
            Y_dnn = dnn.predict(X=X_test)
            accuracy_dnn = accuracy_score(Y_test, Y_dnn)
        else:
            accuracy_dnn = -1.0

        print(
            f"\n\nDataset {dataset['name']}: (d_i={d_i}, d_o={d_o})\nActivation: {a.value}\nAccuracy: \n\tELM:{accuracy_elm * 100:.2f}%\n\tDELM:{accuracy_ffnn * 100:.2f}%\n\tDNN:{accuracy_dnn * 100:.2f}%\n\tCNN:{accuracy_cnn * 100:.2f}%"
        )
    show()
