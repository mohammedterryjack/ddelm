from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from numpy import zeros_like
from copy import deepcopy

from src.utils import Activation
from src.elm import ELM
from src.ffnn import FFNN
from src.cnn import CNN
from src.utils import load_dataset

a = Activation.RELU
d_hs = [20, 10]


for task_name in ("synthetic", "breast_cancer", "iris", "wine", "adult", "diabetes"):
    X, y = load_dataset(name=task_name)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    _, d_i = X.shape
    d_o = max(y) + 1

    elm = ELM(
        input_dimension=d_i,
        output_dimension=d_o,
        hidden_dimension=d_hs[0],
        activation=a,
    )
    ffnn = FFNN(
        input_dimension=d_i,
        output_dimension=d_o,
        hidden_dimensions=d_hs,
        activation=a,
    )
    cnn = CNN(
        input_dimension=d_i,
        output_dimension=d_o,
        kernel_sizes=[(5, 4)],
        hidden_dimensions=d_hs,
        stride=1,
        activation=Activation.RELU,
    )
    dnn = MLPClassifier(hidden_layer_sizes=d_hs, activation=a.name.lower())

    elm.fit(X=X_train, y=Y_train)
    ffnn.fit(X=X_train, y=Y_train)
    cnn.fit(X=X_train, y=Y_train)
    dnn.fit(X=X_train, y=Y_train)

    dnn_from_ffnn = deepcopy(dnn)
    dnn_from_ffnn.coefs_[:-1] = deepcopy(ffnn.Ws[:-1])
    dnn_from_ffnn.intercepts_ = [zeros_like(B) for B in dnn_from_ffnn.intercepts_]
    dnn_from_ffnn.fit(X=X_train, y=Y_train)

    Y_elm = elm.predict(X=X_test)
    Y_ffnn = ffnn.predict(X=X_test)
    Y_cnn = cnn.predict(X=X_test)
    Y_dnn = dnn.predict(X=X_test)
    Y_dnn_from_ffnn = dnn_from_ffnn.predict(X=X_test)

    accuracy_elm = accuracy_score(Y_test, Y_elm)
    accuracy_ffnn = accuracy_score(Y_test, Y_ffnn)
    accuracy_cnn = accuracy_score(Y_test, Y_cnn)
    accuracy_dnn = accuracy_score(Y_test, Y_dnn)
    accuracy_dnn_from_ffnn = accuracy_score(Y_test, Y_dnn_from_ffnn)

    print(
        f"Task: {task_name}\n\tELM:{accuracy_elm * 100:.2f}%\n\tFFNN (novel):{accuracy_ffnn * 100:.2f}%\n\tFFNN (backprop):{accuracy_dnn * 100:.2f}%\n\tFFNN (backprop fine-tuned from novel):{accuracy_dnn_from_ffnn * 100:.2f}%\n\tCNN (novel):{accuracy_cnn * 100:.2f}%"
    )
