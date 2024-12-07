from matplotlib.pyplot import subplots, show
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_diabetes

from src.utils import Activation, inverse_activation
from src.elm import ELM
from src.delm import DELM


for load_data in (load_breast_cancer,):#, load_diabetes):
    data = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size=0.1, random_state=42)

    fig, axes = subplots(len(Activation), 5, figsize=(15, 5)) 
    for j,a in enumerate(Activation): 

        _, d_i = data.data.shape
        d_o = max(data.target) + 1

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

        for i, W in enumerate(delm.Ws):
            ax = axes[j,i]
            ax.imshow(W)  
            if i==0:
                ax.set_ylabel(a.name)
            if j==len(Activation)-1:
                ax.set_xlabel(f"Layer {i}")
        
        Y_elm = elm.predict(X=X_test)
        Y_delm = delm.predict(X=X_test)

        accuracy_elm = accuracy_score(Y_test, Y_elm)
        accuracy_delm = accuracy_score(Y_test, Y_delm)
        print(
            f"\n\nDataset: {data.filename} (d_i={d_i}, d_o={d_o})\nActivation: {a.name}\nAccuracy: \n\tELM:{accuracy_elm * 100:.2f}%\n\tDELM:{accuracy_delm * 100:.2f}%"
        )
    show()
