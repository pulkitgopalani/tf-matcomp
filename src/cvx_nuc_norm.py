import torch
import torch.nn.functional as F
import cvxpy as cp
import numpy as np
import argparse


def nuc_norm_alt(A, mask):
    X = cp.Variable((A.shape[0], A.shape[1]))

    objective = cp.Minimize(cp.norm(X, "nuc"))
    constraints = [cp.multiply(X, mask) == cp.multiply(A, mask)]
    prob = cp.Problem(objective=objective, constraints=constraints)
    result = prob.solve()

    return X.value


def nuc_norm_solver(A, mask):
    X = cp.Variable((A.shape[0], A.shape[1]))

    objective = cp.Minimize(cp.norm(X, "nuc"))
    constraints = [X[mask == 1] == A[mask == 1]]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return X.value


def reg_mse_solver(A, mask, size, reg):
    X = cp.Variable((A.shape[0], A.shape[1]))

    objective = cp.Minimize(
        ((1 / size) * cp.sum_squares(cp.multiply(mask, X - A)))
        + (reg * cp.norm(X, "nuc"))
    )
    prob = cp.Problem(objective=objective)
    result = prob.solve()
    return X.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--r", type=int, default=2)
    parser.add_argument("--mask", type=float, default=0.3)

    args = parser.parse_args()
    n = args.n
    N = args.N
    r = args.r
    p_mask = args.mask

    U = -1 + 2 * torch.rand(size=(N, n, r))
    V = -1 + 2 * torch.rand(size=(N, n, r))

    A_pre = torch.bmm(U, V.permute(0, 2, 1))
    A = torch.round(A_pre, decimals=2).detach().cpu()

    mask = (torch.rand(size=A.shape) > p_mask).cpu().to(float)

    reg_list = [0.0005, 0.001, 0.0015, 0.00175, 0.002, 0.005, 0.01]

    for reg in reg_list:
        sol_reg_mse = torch.zeros_like(A)

        for i in range(A.shape[0]):
            sol_reg_mse[i] = torch.from_numpy(
                reg_mse_solver(
                    A=A[i], mask=mask[i], size=torch.sum(mask[i]).item(), reg=reg
                )
            )

        lobs = F.mse_loss(sol_reg_mse[mask == 1], A[mask == 1]).item()
        lmask = F.mse_loss(sol_reg_mse[mask == 0], A[mask == 0]).item()
        l = F.mse_loss(sol_reg_mse, A).item()

        print(f"${reg}$ &  ${round(lobs,8)}$ & ${round(lmask, 6)}$ & ${round(l, 6)}$\\")
