import argparse
import random

from dotmap import DotMap
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from model import TransformerModel
from data import RealMatrix
import wandb
import numpy as np

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

torch.set_printoptions(precision=4)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"DEVICE: {device}")
torch.set_grad_enabled(False)


plt.rcParams.update(
    {
        "font.size": 24,
        "axes.titlesize": 80,
        "axes.labelsize": 60,
        "xtick.labelsize": 60,
        "ytick.labelsize": 60,
    }
)

plt.rcParams["legend.fontsize"] = 50
plt.rcParams["font.sans-serif"] = "Cantarell"


def main(args):
    m, n, r = args.data.m, args.data.n, args.data.rank
    L, H = args.model.num_hidden_layers, args.model.num_attention_heads

    data_sampler = RealMatrix(args.data, device=device)
    ckpt_dict = torch.load(f"./weights/{n}x{n}_p03_r{r}_l{L}_h{H}_grok49999.tar")

    num_train = args.train.num_train
    num_eval = args.train.num_eval

    X_mask_train, X_train, mask_train = data_sampler.sample(
        n_samples=num_train,
        m=m,
        n=n,
        r=r,
        p_mask=args.data.p_mask,
    )

    X_mask_test, X_test, mask_test = data_sampler.sample(
        n_samples=num_eval, m=m, n=n, r=r, p_mask=args.data.p_mask
    )

    att_mask_train = torch.ones_like(X_mask_train).to(device)
    att_mask_train.requires_grad_(False)
    att_mask_test = torch.ones_like(X_mask_test).to(device)
    att_mask_test.requires_grad_(False)

    # Probe for element
    if args.train.probe_target == "element":
        u_train = X_train.detach().clone().view(-1, 1)
        u_test = X_test.detach().clone().view(-1, 1)

    # Probe for row
    elif args.train.probe_target == "row":
        u_train = torch.empty(num_train, m * n, n).to(device)
        u_test = torch.empty(num_eval, m * n, n).to(device)

        mask_train = mask_train.to(float)
        mask_test = mask_test.to(float)

        for k in range(m * n):
            row_idx = k // n
            u_train[:, k, :] = (X_train * mask_train)[:, row_idx, :].clone().detach()
            u_test[:, k, :] = (X_test * mask_test)[:, row_idx, :].clone().detach()
        u_train = u_train.view(-1, n)
        u_test = u_test.view(-1, n)

    # Probe for singular vector
    elif args.train.probe_target == "svd":
        svd = torch.svd(X_train)[0][:, :, 0]

        u_train = svd.view(-1, 1, m).repeat(1, m * n, 1).view(-1, m)

        u_test = (
            torch.svd(X_test)[0][:, :, 0].view(-1, 1, m).repeat(1, m * n, 1).view(-1, m)
        )

    model = TransformerModel(vocab_size=ckpt_dict["vocab_size"], args=args.model).to(
        device
    )
    model.load_state_dict(ckpt_dict["model"])
    model.eval()

    train_hidden_states = model._backbone(
        input_ids=X_mask_train,
        attention_mask=att_mask_train,
    ).hidden_states

    test_hidden_states = model._backbone(
        input_ids=X_mask_test,
        attention_mask=att_mask_test,
    ).hidden_states

    train_loss, test_loss, train_cosine, test_cosine, mask_loss, obs_loss = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for l in range(1, L + 1):
        layer_train_hidden_states = train_hidden_states[l].view(-1, 768).detach()
        layer_test_hidden_states = test_hidden_states[l].view(-1, 768).detach()

        mask_train = mask_train.view(-1)
        mask_test = mask_test.view(-1)

        probe_model = torch.linalg.lstsq(
            layer_train_hidden_states,
            u_train,
        ).solution

        layer_train_loss = F.mse_loss(
            layer_train_hidden_states @ probe_model,
            u_train,
        ).item()

        layer_test_loss = F.mse_loss(
            layer_test_hidden_states @ probe_model,
            u_test,
        ).item()

        train_loss.append(layer_train_loss)
        test_loss.append(layer_test_loss)

        if args.train.probe_target != "element":
            # remove abs if absolute cosine sim not required
            probe_train_cosine = torch.mean(
                torch.abs(
                    F.cosine_similarity(
                        u_train,
                        layer_train_hidden_states @ probe_model,
                        dim=1,
                    )
                )
            ).item()

            probe_test_cosine = torch.mean(
                torch.abs(
                    F.cosine_similarity(
                        u_test,
                        layer_test_hidden_states @ probe_model,
                        dim=1,
                    )
                )
            ).item()

            print(
                f"Layer {l}:\nTrain MSE: {layer_train_loss} Test MSE: {layer_test_loss} Train cosine: {probe_train_cosine} Test cosine: {probe_test_cosine}"
            )

            train_cosine.append(probe_train_cosine)
            test_cosine.append(probe_test_cosine)
        else:
            mask_test = mask_test.view(-1)
            mask_test_loss = F.mse_loss(
                layer_test_hidden_states[mask_test == 0] @ probe_model,
                u_test[mask_test == 0],
            ).item()

            obs_test_loss = F.mse_loss(
                layer_test_hidden_states[mask_test == 1] @ probe_model,
                u_test[mask_test == 1],
            ).item()

            mask_loss.append(mask_test_loss)
            obs_loss.append(obs_test_loss)

        del probe_model

    fig, ax = plt.subplots(1, 2, figsize=(36, 18))
    ax[0].plot(
        list(range(1, L + 1)),
        train_loss,
        label="Train MSE",
        c="#D1615D",
        linewidth=5,
    )
    ax[0].scatter(
        list(range(1, L + 1)),
        train_loss,
        c="#D1615D",
        linewidth=5,
    )
    ax[0].plot(
        list(range(1, L + 1)), test_loss, label="Test MSE", c="#5778A4", linewidth=5
    )
    ax[0].scatter(list(range(1, L + 1)), test_loss, c="#5778A4", linewidth=5)
    ax[0].set_title("MSE")

    ax[0].set_xlabel("Layer")
    # ax[0].set_ylabel("MSE")
    # ax[0].set_yscale("log")
    ax[0].set_xticks(range(1, 13))
    ax[0].legend(frameon=False)

    if args.train.probe_target != "element":
        ax[1].plot(
            list(range(1, L + 1)),
            train_cosine,
            label="Train",
            c="#D1615D",
            linewidth=5,
        )
        ax[1].scatter(list(range(1, L + 1)), train_cosine, c="#D1615D", linewidth=5)
        ax[1].plot(
            list(range(1, L + 1)),
            test_cosine,
            label="Test",
            c="#5778A4",
            linewidth=5,
        )
        ax[1].scatter(list(range(1, L + 1)), test_cosine, c="#5778A4", linewidth=5)
        ax[1].set_title("Avg. Absolute Cos-Sim")
        ax[1].set_xlabel("Layer")
        # ax[1].set_ylabel("Cosine Similarity")
        ax[1].set_xticks(range(1, 13))
        ax[1].legend(frameon=False)

    else:
        ax[1].plot(
            list(range(1, L + 1)),
            mask_loss,
            label="Masked elements",
            c="#D1615D",
            linewidth=5,
        )
        ax[1].scatter(list(range(1, L + 1)), mask_loss, c="#D1615D", linewidth=5)
        ax[1].plot(
            list(range(1, L + 1)),
            obs_loss,
            label="Observed elements",
            c="#5778A4",
            linewidth=5,
        )
        ax[1].scatter(list(range(1, L + 1)), obs_loss, c="#5778A4", linewidth=5)
        ax[1].set_title("Test MSE")
        ax[1].set_xlabel("Layer")
        # ax[1].set_ylabel("MSE")
        ax[1].set_xticks(range(1, 13))
        ax[1].legend(frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(f"./new_hidden_states/new_probe_L{L}_{args.train.probe_target}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = DotMap(yaml.safe_load(f))

    main(config)

