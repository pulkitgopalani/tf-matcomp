import argparse
from dotmap import DotMap
import yaml
import matplotlib.pyplot as plt

import torch
import numpy as np
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from model import TransformerModel
from data import RealMatrix
import os
from utils import print_output

torch.set_printoptions(precision=4, threshold=100000)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"DEVICE: {device}")

plt.rcParams.update(
    {
        "font.size": 24,
        "axes.titlesize": 80,
        "axes.labelsize": 40,
        "xtick.labelsize": 60,
        "ytick.labelsize": 60,
    }
)

plt.rcParams["legend.fontsize"] = 50
plt.rcParams["font.sans-serif"] = "Cantarell"


def main(args):
    m, n, r = args.data.m, args.data.n, args.data.rank
    L, H = args.model.num_hidden_layers, args.model.num_attention_heads
    epoch = args.train.epoch
    data_sampler = RealMatrix(args.data, device=device)

    ckpt_dict = torch.load(
        os.path.join("./weights", f"{n}x{n}_p03_r2_l{L}_h{H}_grok{epoch}.tar")
    )
    new_epoch = ckpt_dict["epoch"]
    print(f"epoch: {new_epoch} L: {L}")

    token_embed = ckpt_dict["model"]["_backbone.embeddings.word_embeddings.weight"][
        851:1152
    ]

    position_embed = ckpt_dict["model"][
        "_backbone.embeddings.position_embeddings.weight"
    ][: (m * n)]

    if args.train.exp_type == "permute":
        ckpt_dict["model"]["_backbone.embeddings.position_embeddings.weight"][
            : (m * n)
        ] = (
            ckpt_dict["model"]["_backbone.embeddings.position_embeddings.weight"][
                : (m * n)
            ][torch.randperm(m * n)]
            .clone()
            .detach()
        )

        X_mask, X, mask = data_sampler.sample(
            n_samples=args.train.num_eval,
            m=m,
            n=n,
            r=r,
            p_mask=args.data.p_mask,
        )

        X_mask = X_mask.detach().requires_grad_(False)
        X = X.detach().requires_grad_(False)
        mask = mask.detach().cpu().view(mask.shape[0], -1)

        att_mask = torch.ones_like(X_mask).to(device)
        att_mask.requires_grad_(False)

        model = TransformerModel(
            vocab_size=ckpt_dict["vocab_size"], args=args.model
        ).to(device)
        model.load_state_dict(ckpt_dict["model"])
        model.eval()

        model_output = model(X_mask, attention_mask=att_mask).detach().cpu()
        print(
            f"Loss: {F.mse_loss(model_output.cpu(), X.view(model_output.shape).cpu(), reduction='mean').item()}"
        )
        print(
            f"mask Loss: {F.mse_loss(model_output.cpu()[mask==0], X.view(model_output.shape).cpu()[mask==0], reduction='mean').item()}"
        )
        print(
            f"obs Loss: {F.mse_loss(model_output.cpu()[mask==1], X.view(model_output.shape).cpu()[mask==1], reduction='mean').item()}"
        )
        print_output(0, 0, 0, m, n, r, output=model_output[0], X=X[0], mask=mask[0])

    elif args.train.exp_type == "token_norm":
        plt.figure(figsize=(12, 10))
        plt.plot(
            np.arange(-1.5, 1.51, 0.01),
            torch.linalg.norm(token_embed, dim=-1).detach().cpu().numpy(),
        )


        plt.xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
        plt.xlabel("value")
        plt.ylabel(r"$\ell_2$ norm")
        plt.tight_layout()
        plt.savefig(f"./new_hidden_states/token_norm_{new_epoch}.pdf")

    elif args.train.exp_type == "token_pca":
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        plt.figure(figsize=(12, 10))

        proj_hidden_state = PCA(n_components=2)

        hidden_state_embed = proj_hidden_state.fit_transform(
            token_embed.view(-1, 768).detach().cpu().numpy()
        )
        for i in range(5):

            plt.scatter(
                hidden_state_embed[50 * i : 50 * (i + 1), 0],
                hidden_state_embed[50 * i : 50 * (i + 1), 1],
                label=f"[{0.5*i - 1.5}, {0.5*i - 1})",
                color=colors[i],
                linewidth=5,
            )

        plt.scatter(
            hidden_state_embed[250:301, 0],
            hidden_state_embed[250:301, 1],
            label="[1, 1.5]",
            color=colors[5],
            linewidth=5,
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./new_hidden_states/token_pca_{new_epoch}.pdf")

    elif args.train.exp_type == "pos_norm":
        plt.figure(figsize=(12, 10))
        plt.plot(
            np.arange(0, 49),
            torch.linalg.norm(position_embed, dim=-1).detach().cpu().numpy(),
        )
        plt.xticks(
            np.arange(0, 49, 7),
        )
        plt.xlabel("position")
        plt.ylabel("L2 norm")
        plt.tight_layout()
        plt.savefig(f"./new_hidden_states/pos_norm_{new_epoch}.pdf")

    elif args.train.exp_type == "pos_tsne":
        plt.figure(figsize=(15, 12))
        proj_hidden_state = TSNE(n_components=2)
        hidden_state_embed = proj_hidden_state.fit_transform(
            position_embed.view(-1, 768).detach().cpu().numpy()
        )

        for i in range(7):
            plt.scatter(
                hidden_state_embed[i::7, 0],
                hidden_state_embed[i::7, 1],
                label=f"Col {i+1}",
                linewidth=10,
            )

        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(f"./new_hidden_states/pos_tsne_{new_epoch}.pdf")

    elif args.train.exp_type == "token_progress":
        colors = plt.cm.viridis(np.linspace(0, 1, 6))

        final_ckpt = torch.load(
            os.path.join("./weights", f"{n}x{n}_p03_r2_l{L}_h{H}_grok49999.tar")
        )
        final_token_embed = final_ckpt["model"][
            "_backbone.embeddings.word_embeddings.weight"
        ][851:1152]

        pca_hidden_state = PCA(n_components=2)
        _ = pca_hidden_state.fit_transform(
            final_token_embed.view(-1, 768).detach().cpu().numpy()
        )
        final_pc = pca_hidden_state.components_[:2]

        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(75, 12))
        for step, pre_epoch in enumerate([3999, 9999, 13999, 15999, 19999]):

            pre_ckpt_dict = torch.load(
                os.path.join(
                    "./weights", f"{n}x{n}_p03_r2_l{L}_h{H}_grok{pre_epoch}.tar"
                )
            )

            pre_embeds = pre_ckpt_dict["model"][
                "_backbone.embeddings.word_embeddings.weight"
            ][851:1152]

            print(f"pre epoch: {pre_ckpt_dict['epoch']}")

            hidden_state_embed = pre_embeds.detach().cpu().numpy() @ final_pc.T
            for i in range(5):
                ax[step].scatter(
                    hidden_state_embed[50 * i : 50 * (i + 1), 0],
                    hidden_state_embed[50 * i : 50 * (i + 1), 1],
                    label=f"[{0.5*i - 1.5}, {0.5*i - 1})",
                    color=colors[i],
                )

            ax[step].scatter(
                hidden_state_embed[250:301, 0],
                hidden_state_embed[250:301, 1],
                label="[1, 1.5]",
                color=colors[5],
            )
            ax[step].set_title(f"Step {pre_epoch+1}")

        plt.tight_layout()
        plt.savefig(f"./new_hidden_states/token_progress.pdf")
        plt.close()

    elif args.train.exp_type == "pos_progress":
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(80, 16))

        for step, pre_epoch in enumerate([3999, 9999, 13999, 15999, 19999]):
            ckpt_dict = torch.load(
                os.path.join(
                    "./weights", f"{n}x{n}_p03_r2_l{L}_h{H}_grok{pre_epoch}.tar"
                )
            )

            position_embed = ckpt_dict["model"][
                "_backbone.embeddings.position_embeddings.weight"
            ][: (m * n)]

            proj_hidden_state = TSNE(n_components=2)
            hidden_state_embed = proj_hidden_state.fit_transform(
                position_embed.view(-1, 768).detach().cpu().numpy()
            )

            for col in range(7):
                ax[step].scatter(
                    hidden_state_embed[col::7, 0],
                    hidden_state_embed[col::7, 1],
                    label=f"Column {col+1}",
                    linewidth=10,
                )

            ax[step].set_title(f"Step {pre_epoch+1}")

        plt.tight_layout()
        plt.savefig(f"./new_hidden_states/pos_progress.pdf")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = DotMap(yaml.safe_load(f))

    main(config)
