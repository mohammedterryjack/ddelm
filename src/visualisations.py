from matplotlib.pyplot import subplots, show, annotate, Normalize, cm
from numpy import ndarray
from numpy.linalg import pinv

from src.cnn import CNN
from src.ffnn import FFNN
from src.utils import one_hot_encode

SUBSCRIPTS = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]


def display_backward_pass_ffnn(model:FFNN, X:ndarray, Y:ndarray) -> None:
    n_rows = len(model.Ws)
    n_cols = 2 * len(model.Ws) + 2
    _, axes = subplots(
        n_rows,
        n_cols,
        figsize=(15, 5),
    )

    y_expected = one_hot_encode(class_ids=Y, n_classes=model.d_o)
    
    for i in range(len(model.Ws)):
        
        Y_hat_next = model.backward(
            Y=y_expected, 
            Ws=model.Ws[i + 1 :], 
            inverse_activation=model.inverse_activation
        ) if i < len(model.Ws) else Y_expected
        Y_hat = model.forward(
            X=X, 
            Ws=model.Ws[:i], 
            activation=model.activation
        )
        model.Ws[i] = pinv(Y_hat) @ Y_hat_next

        for j in range(len(model.Ws)):
            axes[i,1+2*j].imshow(model.Ws[j])
    
    #axes[j].imshow(y_predicted, cmap="pink")
    #axes[j].set_title(f"Ŷ{SUBSCRIPTS[i]} = Ŷₒᵤₜ")
    #axes[j + 1].imshow(y_expected, cmap="YlGn_r")
    #axes[j + 1].set_title("Y")
    show()


def display_forward_pass_cnn(model: CNN, X: ndarray, Y: ndarray) -> None:
    y_expected = one_hot_encode(class_ids=Y, n_classes=model.d_o)
    y_predicted = one_hot_encode(class_ids=model.predict(X=X), n_classes=model.d_o)

    n_cols = 3 * len(model.Wks) + 2 * len(model.Whs) + 1
    fig, axes = subplots(
        2,
        n_cols,
        figsize=(15, 5),
    )

    n_weights = len(model.Wks + model.Whs)
    row = 0
    for i, W in enumerate(model.Wks + model.Whs):

        if i >= len(model.Wks):  # ffnn layers
            Y_hat = model.forward_pass_ffnn(
                X=model.forward_pass_cnn(
                    X=X, Ws=model.Wks, activation=model.activation, stride=model.stride
                ),
                Ws=model.Whs[: i - len(model.Wks)],
                activation=model.activation,
            )

            axes[1, row].axis("off")
            axes[1, row + 1].axis("off")

            axes[0, row].imshow(Y_hat, cmap="pink")
            axes[0, row + 1].imshow(W, cmap="inferno")

            axes[0, row].set_title(
                f"Ŷ{SUBSCRIPTS[i+1]} = f⁻¹(H{SUBSCRIPTS[i+1]})"
                if i == len(model.Wks)
                else f"Ŷ{SUBSCRIPTS[i+1]} = H{SUBSCRIPTS[i+1]}"
            )
            axes[0, row + 1].set_title(f"W{SUBSCRIPTS[i+1]}")

            box1 = axes[0, row].get_position()
            box2 = axes[0, row + 1].get_position()
            box3 = axes[0, row + 2].get_position()

            annotate(
                "",
                xy=(box2.x0, (box2.y0 + box2.y1) / 2),
                xytext=(box1.x1, (box1.y0 + box1.y1) / 2),
                xycoords="figure fraction",
                arrowprops=dict(arrowstyle="-", color="black", lw=1),
            )
            annotate(
                "α",
                xy=(box3.x0, (box3.y0 + box3.y1) / 2),
                xytext=(box2.x1, (box2.y0 + box2.y1) / 2),
                xycoords="figure fraction",
                arrowprops=dict(arrowstyle="->", color="red", lw=1),
            )

        else:  # cnn layers
            Y_hat = model.forward_pass_cnn(
                X=X, Ws=model.Wks[:i], activation=model.activation, stride=model.stride
            )
            Y_hat_cnn = model.forward_pass_ff_to_cnn_layer(
                ff_layer=Y_hat,
                window_size=W.shape[0],
                stride=model.stride,
            )
            Y_hat_next_cnn = model.activation(Y_hat_cnn @ W)
            Y_hat_next = model.forward_pass_cnn_to_ff_layer(cnn_layer=Y_hat_next_cnn)

            axes[0, row + 1].axis("off")
            axes[1, row].axis("off")
            axes[1, row + 2].axis("off")

            axes[1, row] = fig.add_subplot(2, n_cols - 1, n_cols + row, projection="3d")
            axes[1, row + 2] = fig.add_subplot(
                2, n_cols - 1, n_cols + row + 2, projection="3d"
            )

            axes[0, row].imshow(Y_hat, cmap="pink")
            axes[1, row].voxels(
                Y_hat_cnn,
                facecolors=cm.pink(
                    Normalize(Y_hat_cnn.min(), Y_hat_cnn.max())(Y_hat_cnn)
                ),
            )
            axes[1, row + 1].imshow(W, cmap="inferno")
            axes[1, row + 2].voxels(
                Y_hat_next_cnn,
                facecolors=cm.pink(
                    Normalize(Y_hat_next_cnn.min(), Y_hat_next_cnn.max())(
                        Y_hat_next_cnn
                    )
                ),
            )
            axes[0, row + 2].imshow(Y_hat_next, cmap="pink")

            axes[0, row].set_title(f"Ŷ{SUBSCRIPTS[i+1]} = X" if i == 0 else "")
            axes[1, row].set_title(f"f(Ŷ{SUBSCRIPTS[i+1]})")
            axes[1, row + 1].set_title(f"W{SUBSCRIPTS[i+1]}")
            axes[1, row + 2].set_title(f"H{SUBSCRIPTS[i+2]}")
            axes[0, row + 2].set_title(f"Ŷ{SUBSCRIPTS[i+2]} = f⁻¹(H{SUBSCRIPTS[i+2]})")

            box1 = axes[0, row].get_position()
            box2 = axes[1, row].get_position()
            box3 = axes[1, row + 1].get_position()
            box4 = axes[1, row + 2].get_position()
            box5 = axes[0, row + 2].get_position()

            annotate(
                "",
                xytext=((box1.x0 + box1.x1) / 2, box1.y0),
                xy=((box2.x0 + box2.x1) / 2, box2.y1),
                xycoords="figure fraction",
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
            )
            annotate(
                "",
                xytext=(box2.x1, (box2.y0 + box2.y1) / 2),
                xy=(box3.x0, (box3.y0 + box3.y1) / 2),
                xycoords="figure fraction",
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
            )
            annotate(
                "α",
                xytext=(box3.x1, (box3.y0 + box3.y1) / 2),
                xy=(box4.x0, (box4.y0 + box4.y1) / 2),
                xycoords="figure fraction",
                arrowprops=dict(arrowstyle="->", color="red", lw=1),
            )
            annotate(
                "",
                xytext=((box4.x0 + box4.x1) / 2, box4.y1),
                xy=((box5.x0 + box5.x1) / 2, box5.y0),
                xycoords="figure fraction",
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
            )

        row += 2 + int(i < len(model.Wks) - 1)

    axes[1, row].axis("off")
    axes[1, row + 1].axis("off")

    axes[0, row].imshow(y_predicted, cmap="pink")
    axes[0, row + 1].imshow(y_expected, cmap="YlGn_r")

    axes[0, row].set_title(f"Ŷ{SUBSCRIPTS[n_weights+1]} = Ŷₒᵤₜ")
    axes[0, row + 1].set_title("Y")

    show()


def display_forward_pass_ffnn(model: FFNN, X: ndarray, Y: ndarray) -> None:

    y_expected = one_hot_encode(class_ids=Y, n_classes=model.d_o)
    y_predicted = one_hot_encode(class_ids=model.predict(X=X), n_classes=model.d_o)

    _, axes = subplots(1, 2 * len(model.Ws) + 2, figsize=(15, 5))
    for i in range(len(model.Ws)):
        Y_hat = model.forward(X=X, Ws=model.Ws[:i], activation=model.activation)
        j = 2 * i
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
