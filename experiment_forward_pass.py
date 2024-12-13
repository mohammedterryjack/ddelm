from sklearn.datasets import load_breast_cancer
from matplotlib.pyplot import subplots, show

from src.utils import Activation, one_hot_encode
from src.delm import DELM

data = load_breast_cancer()

X = data.data
Y = data.target

_, d_i = X.shape
d_o = max(Y) + 1

y = one_hot_encode(class_ids=Y, n_classes=d_o)

X_test = X[:100]
y_test = y[:100]

delm = DELM(
    input_dimension=d_i, output_dimension=d_o, 
    hidden_dimensions=[10,5,6,8],
    activation=Activation.RELU
)
delm.fit(X=X, y=Y)

Y_hat = delm.predict(X=X_test)
y_predicted = one_hot_encode(class_ids=Y_hat, n_classes=d_o)

subscripts = ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"]
_, axes = subplots(1, 2*len(delm.Ws)+2, figsize=(15, 5)) 
for i in range(len(delm.Ws)):
    j = 2*i
    Y_hat = delm.forward(
        X=X_test,
        Ws=delm.Ws[:i],
        activation=delm.activation
    )
    axes[j].imshow(Y_hat,cmap='pink')
    axes[j].set_title(f"X = Ŷ{subscripts[i]}" if j == 0 else f"h{subscripts[i]} = Ŷ{subscripts[i]}")
    axes[j+1].imshow(delm.Ws[i], cmap='inferno')
    axes[j+1].set_title("Wᵢₙ" if i==0 else "Wₒᵤₜ" if i==len(delm.Ws)-1 else f"Wₕ{subscripts[i]}")
i = len(delm.Ws)
j = 2*i
axes[j].imshow(y_predicted,cmap='pink')
axes[j].set_title(f"Ŷₒᵤₜ = Ŷ{subscripts[i]}")
axes[j+1].imshow(y_test,cmap='YlGn_r')
axes[j+1].set_title("Y")
show()