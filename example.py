from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris

from src.utils import Activation, inverse_activation
from src.elm import ELM
from src.delm import DELM

for load_data in (load_breast_cancer, load_iris):
    data = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.5, random_state=42)

    _, d_i = data.data.shape
    d_o = max(data.target) + 1

    for a in (Activation.SINE, Activation.RELU):
        elm = ELM(
            input_dimension=d_i, output_dimension=d_o, 
            hidden_dimension=100,
            activation=a
        )
        delm = DELM(
            input_dimension=d_i, output_dimension=d_o, 
            hidden_dimensions=[40, 30, 20, 10],
            activation=a
        )

        elm.fit(X=X_train, Y=Y_train)
        delm.fit(X=X_train, Y=Y_train)

        Y_elm = elm.predict(X=X_test)
        Y_delm = delm.predict(X=X_test)

        accuracy_elm = accuracy_score(Y_test, Y_elm)
        accuracy_delm = accuracy_score(Y_test, Y_delm)
        print(
            f"\n\nDataset: {data.filename}\nActivation: {a.name}\nAccuracy: \n\tELM:{accuracy_elm * 100:.2f}%\n\tDELM:{accuracy_delm * 100:.2f}%"
        )
