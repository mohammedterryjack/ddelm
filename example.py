from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer as load_data  ###load_iris  load_wine

from src.elm import ELM
from src.delm import DELM

data = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.9, random_state=42)

_, d_i = data.data.shape
d_o = max(data.target) + 1


elm = ELM(input_dimension=d_i, output_dimension=d_o, hidden_dimension=100)
delm = DELM(
    input_dimension=d_i, output_dimension=d_o, hidden_dimensions=[40, 30, 20, 10]
)

elm.fit(X=X_train, Y=Y_train)
delm.fit(X=X_train, Y=Y_train)

Y_elm = elm.predict(X=X_test)
Y_delm = delm.predict(X=X_test)

accuracy_elm = accuracy_score(Y_test, Y_elm)
accuracy_delm = accuracy_score(Y_test, Y_delm)
print(
    f"Accuracy: \n\tELM:{accuracy_elm * 100:.2f}%\n\tDELM:{accuracy_delm * 100:.2f}%"
)
