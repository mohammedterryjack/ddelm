from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification,
    fetch_openml
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from numpy import hstack, ndarray
from matplotlib.pyplot import subplots, show

from src.utils import Activation
from src.elm import ELM
from src.ffnn import FFNN

def get_openml_dataset(name:str,version:int) -> tuple[ndarray,ndarray]:
    scaler = StandardScaler()
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    data = fetch_openml(name,version=version)
    x = data.data
    y = data.target

    numerical_column_names = x.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_column_names = x.select_dtypes(include=['object', 'category']).columns.tolist()
    return (
        hstack([scaler.fit_transform(x[numerical_column_names]), onehot_encoder.fit_transform(x[categorical_column_names])]),
        y.cat.codes.to_numpy()
    )


a = Activation.RELU
d_hs = [30,20]
tasks = {
    "synthetic":make_classification,
    "breast cancer":lambda : (
        load_breast_cancer().data,
        load_breast_cancer().target 
    ),
    "iris":lambda :(
        load_iris().data,
        load_iris().target,
    ),
    "wine":lambda :(
        load_wine().data,
        load_wine().target
    ),
    "adult":lambda:get_openml_dataset(name="adult", version=2),
    "diabetes":lambda:get_openml_dataset(name="diabetes", version=1),
}

_, axes = subplots(len(tasks),4,figsize=(15, 5))
axes[0,0].set_title("Target")
axes[0,1].set_title("ELM")
axes[0,2].set_title("FFNN (novel)")
axes[0,3].set_title("FFNN (backprop)")


for i,(task_name,load_dataset) in enumerate(tasks.items()):
    X,y = load_dataset()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    _, d_i = X.shape
    d_o = max(y) + 1

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
    dnn = MLPClassifier(
        hidden_layer_sizes=d_hs, 
        activation=a.name.lower()
    )

    elm.fit(X=X_train, y=Y_train)
    ffnn.fit(X=X_train, y=Y_train)    
    dnn.fit(X=X_train, y=Y_train)
    
    Y_elm = elm.predict(X=X_test)
    Y_ffnn = ffnn.predict(X=X_test)
    Y_dnn = dnn.predict(X=X_test)

    accuracy_elm = accuracy_score(Y_test, Y_elm)
    accuracy_ffnn = accuracy_score(Y_test, Y_ffnn)
    accuracy_dnn = accuracy_score(Y_test, Y_dnn)
    print(f"Task: {task_name}\n\tELM:{accuracy_elm * 100:.2f}%\n\tFFNN:{accuracy_ffnn * 100:.2f}%\n\tDNN:{accuracy_dnn * 100:.2f}%")    

    #axes[i,0].scatter(X_train[:,0], X_train[:, 1], c=Y_train)
    #axes[i,1].scatter(X_train[:,0], X_train[:, 1], c=Y_train)
    #axes[i,2].scatter(X_train[:,0], X_train[:, 1], c=Y_train)
    #axes[i,3].scatter(X_train[:,0], X_train[:, 1], c=Y_train)
    
    axes[i,0].scatter(X_test[:, 0], X_test[:, 1], c=Y_test, alpha=0.6)
    axes[i,1].scatter(X_test[:, 0],X_test[:, 1],c=Y_elm,alpha=0.6)
    axes[i,2].scatter(X_test[:, 0],X_test[:, 1],c=Y_ffnn,alpha=0.6)
    axes[i,3].scatter(X_test[:, 0],X_test[:, 1],c=Y_dnn,alpha=0.6)


show()