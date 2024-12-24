from enum import Enum
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification,
    fetch_openml,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from numpy import ndarray, zeros, maximum, nan_to_num, tan, arctan, hstack


class Activation(Enum):
    IDENTITY = "identity"
    RELU = "rectified linear unit"
    TAN = "tan"


def activation_function(activation: Activation) -> callable:
    return {
        Activation.IDENTITY: lambda x: x,
        Activation.RELU: lambda x: maximum(0, x),
        Activation.TAN: tan,
    }[activation]


def inverse_activation(activation: Activation) -> callable:
    f = {
        Activation.TAN: arctan,
    }.get(activation, lambda x: x)
    return lambda x: nan_to_num(f(x), nan=0.0)


def one_hot_encode(class_ids: list[int], n_classes: int) -> ndarray:
    n_samples = len(class_ids)
    vectors = zeros(shape=(n_samples, n_classes), dtype=int)
    vectors[range(n_samples), class_ids] = 1
    return vectors


def get_openml_dataset(name: str, version: int) -> tuple[ndarray, ndarray]:
    scaler = StandardScaler()
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    data = fetch_openml(name, version=version)
    x = data.data
    y = data.target

    numerical_column_names = x.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_column_names = x.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return (
        hstack(
            [
                scaler.fit_transform(x[numerical_column_names]),
                onehot_encoder.fit_transform(x[categorical_column_names]),
            ]
        ),
        y.cat.codes.to_numpy(),
    )


def get_sklearn_dataset(name: str) -> tuple[ndarray, ndarray]:
    data = {"iris": load_iris, "breast_cancer": load_breast_cancer, "wine": load_wine}[
        name
    ]()
    x = data.data
    y = data.target
    return x, y


def load_dataset(name: str) -> tuple[ndarray, ndarray]:
    return {
        "synthetic": make_classification,
        "breast_cancer": lambda: get_sklearn_dataset(name=name),
        "iris": lambda: get_sklearn_dataset(name=name),
        "wine": lambda: get_sklearn_dataset(name=name),
        "adult": lambda: get_openml_dataset(name=name, version=2),
        "diabetes": lambda: get_openml_dataset(name=name, version=1),
    }[name]()
