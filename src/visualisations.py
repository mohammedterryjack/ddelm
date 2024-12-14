from matplotlib.pyplot import subplots, show, annotate
from numpy import ndarray

from src.delm import DELM
from src.utils import one_hot_encode

SUBSCRIPTS = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]


def display_forward_pass_ffnn(model: DELM, X: ndarray, Y: ndarray) -> None:

    y_expected = one_hot_encode(class_ids=Y, n_classes=model.d_o)
    y_predicted = one_hot_encode(class_ids=model.predict(X=X), n_classes=model.d_o)

    _, axes = subplots(1, 2 * len(model.Ws) + 2, figsize=(15, 5))
    for i in range(len(model.Ws)):
        j = 2 * i
        Y_hat = model.forward(X=X, Ws=model.Ws[:i], activation=model.activation)
        axes[j].imshow(Y_hat, cmap="pink")
        axes[j].set_title(
            f"Ŷ{SUBSCRIPTS[i]} = X"
            if j == 0
            else f"Ŷ{SUBSCRIPTS[i]} = H{SUBSCRIPTS[i]}"
        )
        axes[j + 1].imshow(model.Ws[i], cmap="inferno")
        axes[j + 1].set_title(
            "Wᵢₙ"
            if i == 0
            else "Wₒᵤₜ" if i == len(model.Ws) - 1 else f"W{SUBSCRIPTS[i]}"
        )

        box1 = axes[j + 1].get_position()
        box2 = axes[j + 2].get_position()

        annotate(
            "α",
            xy=(box2.x0, (box2.y0 + box2.y1) / 2),
            xytext=(box1.x1, (box1.y0 + box1.y1) / 2),
            xycoords="figure fraction",
            arrowprops=dict(arrowstyle="->", color="red", lw=1),
        )

    i = len(model.Ws)
    j = 2 * i
    axes[j].imshow(y_predicted, cmap="pink")
    axes[j].set_title(f"Ŷ{SUBSCRIPTS[i]} = Ŷₒᵤₜ")
    axes[j + 1].imshow(y_expected, cmap="YlGn_r")
    axes[j + 1].set_title("Y")
    show()
