import numpy as np
import cvxpy as cp


def print_subspace_output(
    epoch, train_loss, eval_loss, L, r, output, X, mask  # eval_nuc_norm_loss
):
    print(
        f"({L}, {r}) Step {epoch} -- train loss: {round(train_loss, 4)}, eval loss: {round(eval_loss, 4)}"
    )
    out = []
    for i in range(X.shape[0]):
        out.append(
            f"({round(float(output.view(X.shape)[i]), 4)}, {round(X[i].item(), 4)}, {int(mask.view(X.shape)[i])})"
        )
    print(" ".join(out))
    print("------------")


def print_output(
    epoch,
    train_loss,
    eval_loss,
    m,
    n,
    r,
    output,
    X,
    mask,
    mask_loss=None,
    obs_loss=None,
    mean_mask_value=None,
):
    out = []
    if (
        (mask_loss is not None)
        and (obs_loss is not None)
        and (mean_mask_value is not None)
    ):
        print(
            f"({m}, {n}, {r}) Step {epoch} -- train loss: {round(train_loss, 4)}, eval loss: {round(eval_loss, 4)}, mask loss: {round(mask_loss, 4)}, obs loss: {round(obs_loss, 4)}, mask value: {round(mean_mask_value, 4)}"
        )  # , nuc norm: {round(eval_nuc_norm_loss, 4)}"
    else:
        print(
            f"({m}, {n}, {r}) Step {epoch} -- train loss: {round(train_loss, 4)}, eval loss: {round(eval_loss, 4)}"
        )

    for i in range(X.shape[0]):
        out = []
        for j in range(X.shape[1]):
            out.append(
                f"({round(float(output.view(X.shape)[i,j]), 4)}, {round(X[i,j].item(), 4)}, {int(mask.view(X.shape)[i,j] != 0)})"
            )
        print(f"{' '.join(out)}")
    print("------------")
