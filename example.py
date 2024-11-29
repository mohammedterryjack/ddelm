from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer as load_data ##load_wine#load_iris

from src.elm import ELM 
from src.ddelm import DDELM 

data = load_data()
X = data.data 
Y_true = data.target 
X_train, _, Y_train,_ = train_test_split(X, Y_true, test_size=0.3, random_state=42)

_,d_i = X.shape
d_o = max(data.target)+1



elm = ELM(input_dimension=d_i,output_dimension=d_o,hidden_dimension=150)
elm.fit(X=X_train,Y=Y_train)
Y_elm = elm.predict(X=X)

 
ddelm = DDELM(
    input_dimension=d_i,output_dimension=d_o,
    hidden_dimensions=[50,40,30,20,10]
)
ddelm.fit(X=X_train,Y=Y_train)
Y_ddelm = ddelm.predict(X=X)

 
accuracy_elm = accuracy_score(Y_true, Y_elm)
accuracy_ddelm = accuracy_score(Y_true, Y_ddelm)
print(f"Accuracy: \n\tELM:{accuracy_elm * 100:.2f}%\n\tDDELM:{accuracy_ddelm * 100:.2f}%")