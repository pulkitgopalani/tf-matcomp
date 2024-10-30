import argparse
from dotmap import DotMap
import yaml
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F


from model import TransformerModel
from data import RealMatrix
from cvx_nuc import nuc_norm_solver

torch.set_printoptions(precision=4)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")
# plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})

plt.rcParams.update(
    {
        "font.size": 24,
        "axes.titlesize": 60,
        "axes.labelsize": 60,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
    }
)

plt.rcParams["legend.fontsize"] = 45
plt.rcParams["font.sans-serif"] = "Cantarell"

def main(args):
    m, n, r, epoch = args.data.m, args.data.n, args.data.rank, args.train.epoch
    L, H = args.model.num_hidden_layers, args.model.num_attention_heads

    data_sampler = RealMatrix(args.data, device=device)

    model_path = f"{n}x{n}_p03_r2_l{L}_h{H}_grok{epoch}.tar"

    cvx_norms, cvx_mse = [], []
    tf_norms, tf_mse = [], []

    fig, ax = plt.subplots(1, 2, figsize=(22, 15))

    p_mask_list = np.arange(start=0.1, stop=0.64, step=0.05)
    for p_mask in p_mask_list:
        X_mask, X, mask = data_sampler.sample(
            n_samples=args.train.num_eval,
            m=m,
            n=n,
            r=r,
            p_mask=p_mask,
        )

        X_mask = X_mask.detach().requires_grad_(False)
        X = X.detach().requires_grad_(False)

        att_mask = torch.ones_like(X_mask).to(device)
        att_mask.requires_grad_(False)

        nuc_norm_outputs = torch.zeros_like(X).to("cpu")
        for i in range(X.shape[0]):
            nuc_norm_outputs[i] = torch.from_numpy(
                nuc_norm_solver(
                    A=X[i].detach().cpu().numpy(), mask=mask[i].detach().cpu().to(float).numpy()
                )
            )

        cvx_norms.append(
            torch.mean(
                torch.linalg.matrix_norm(nuc_norm_outputs, ord="nuc", dim=(-2, -1))
            ).item()
        )
        cvx_mse.append(
            F.mse_loss(
                input=nuc_norm_outputs,
                target=X.to("cpu"),
                reduction="mean",
            ).item()
        )

        ckpt_dict = torch.load(os.path.join("./weights", model_path))
        model = TransformerModel(
            vocab_size=ckpt_dict["vocab_size"], args=args.model
        ).to(device)
        model.load_state_dict(ckpt_dict["model"])
        model.eval()
        print(ckpt_dict["epoch"])
        model_output = (
            model(X_mask, attention_mask=att_mask).view(X.shape).detach().cpu()
        )

        tf_mse.append(
            F.mse_loss(input=model_output, target=X.to("cpu"), reduction="mean").item()
        )

        tf_norms.append(
            torch.mean(
                torch.linalg.matrix_norm(model_output, ord="nuc", dim=(-2, -1))
            ).item()
        )

    ax[0].plot(p_mask_list, cvx_mse, c="#D1615D", label="nuclear norm min", linewidth=5)
    ax[0].scatter(p_mask_list, cvx_mse, c="#D1615D", linewidth=7)

    ax[0].plot(p_mask_list, tf_mse, c="#5778A4", label="BERT", linewidth=5)
    ax[0].scatter(p_mask_list, tf_mse, c="#5778A4", linewidth=7)

    ax[1].plot(p_mask_list, cvx_norms, c="#D1615D", label="nuclear norm min", linewidth=5)
    ax[1].scatter(p_mask_list, cvx_norms, c="#D1615D", linewidth=7)

    ax[1].plot(p_mask_list, tf_norms, c="#5778A4", label="BERT", linewidth=5)
    ax[1].scatter(p_mask_list, tf_norms, c="#5778A4", linewidth=7)

    ax[0].set_title("MSE")
    ax[1].set_title("Nuclear Norm")

    ax[0].set_xlabel(r"$p_{\rm mask}$")
    #ax[0].set_ylabel("MSE")
    ax[0].set_yscale("log")

    ax[1].set_xlabel(r"$p_{\rm mask}$")
    ax[1].set_yscale("log")

    ax[0].legend(frameon=False)
    #ax[1].legend(frameon=False)

    plt.tight_layout()
    plt.savefig(f"./new_hidden_states/compare_m{m}_n{n}_r{r}_l{L}_h{H}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = DotMap(yaml.safe_load(f))

    # print(config)
    main(config)
