from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from src.utils import Activation
from src.elm import ELM
from src.ffnn import FFNN
from src.cnn import CNN

data = load_digits()

X_train, X_test, Y_train, Y_test = train_test_split(
    data.data, data.target, test_size=0.01, random_state=42
)

d_hs = [100, 80, 70]
for j, a in enumerate(Activation):
    _, d_i = data.data.shape
    d_o = max(data.target) + 1

    elm = ELM(
        input_dimension=d_i,
        output_dimension=d_o,
        hidden_dimension=sum(d_hs),
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
        hidden_dimensions=[100, 30],
        kernel_sizes=[(5, 4), (7, 6)],
        stride=1,
        activation=a,
    )
    
    elm.fit(X=X_train, y=Y_train)
    ffnn.fit(X=X_train, y=Y_train)
    cnn.fit(X=X_train, y=Y_train)
       
    Y_elm = elm.predict(X=X_test)
    Y_ffnn = ffnn.predict(X=X_test)
    Y_cnn = cnn.predict(X=X_test)

    accuracy_elm = accuracy_score(Y_test, Y_elm)
    accuracy_ffnn = accuracy_score(Y_test, Y_ffnn)
    accuracy_cnn = accuracy_score(Y_test, Y_cnn)
    print(f"Activation: {a.value}\n\tELM:{accuracy_elm * 100:.2f}%\n\tFFNN:{accuracy_ffnn * 100:.2f}%\n\tCNN:{accuracy_cnn * 100:.2f}%")    

