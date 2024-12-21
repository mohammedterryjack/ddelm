from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification
)
from sklearn.neural_network import MLPClassifier

from src.utils import Activation
from src.elm import ELM
from src.ffnn import FFNN
from src.cnn import CNN

#NLTK Datasets
    #Reuters
    #Movie Reviews
    #20 Newsgroups
#Hugging Face Datasets
    #AG News
    #IMDb
    #TREC
    #Yahoo Answers
    #TweetEval
    

a = Activation.RELU
d_hs = [3,2]
tasks = {
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
    "synthetic":make_classification
}

for task_name,load_dataset in tasks.items():
    X,y = load_dataset()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.01, random_state=42
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
