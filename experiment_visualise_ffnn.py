from sklearn.datasets import load_breast_cancer

from src.utils import Activation, one_hot_encode
from src.delm import DELM
from src.visualisations import display_forward_pass_ffnn

data = load_breast_cancer()
delm = DELM(
    input_dimension=data.data.shape[1],
    output_dimension=max(data.target) + 1,
    hidden_dimensions=[10, 5, 6, 8],
    activation=Activation.RELU,
)
display_forward_pass_ffnn(model=delm, X=data.data[:100], Y=data.target[:100])
delm.fit(X=data.data, y=data.target)
display_forward_pass_ffnn(model=delm, X=data.data[:100], Y=data.target[:100])
