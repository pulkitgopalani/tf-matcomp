import argparse
from dotmap import DotMap
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import numpy as np
import torch.nn.functional as F
from model import TransformerModel
from data import RealMatrix
import os
from utils import print_output

torch.set_printoptions(precision=4, threshold=100000)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"DEVICE: {device}")


def main(args):
    m, n, r = args.data.m, args.data.n, args.data.rank
    data_sampler = RealMatrix(args.data, device=device)
    L, H = args.model.num_hidden_layers, args.model.num_attention_heads

    X_mask_test, X_test, mask_test = data_sampler.sample(
        n_samples=args.train.num_eval,
        m=m,
        n=n,
        r=r,
        p_mask=args.data.p_mask,
        return_uv=False,
    )

    X_mask_test = X_mask_test.detach().requires_grad_(False)
    X_test = X_test.detach().requires_grad_(False)

    test_att = torch.ones_like(X_mask_test).requires_grad_(False)

    mask_type, mask1, mask2, obs1, obs2 = (
        args.train.mask_type,
        args.train.mask1,
        args.train.mask2,
        args.train.obs1,
        args.train.obs2,
    )

    if mask_type != "none":
        mask_test = (
            torch.ones(size=(args.train.num_eval, m, n)).detach().requires_grad_(False)
        ).to(device).to(int)

        if mask_type == "row":
            mask_test[:, mask1, :] = 0
            mask_test[:, mask2, :] = 0
            mask_test[:, mask1, obs1] = 1
            mask_test[:, mask2, obs2] = 1

    X_mask_test = data_sampler.tokenize(X_test, mask_test)[0]

    epoch = args.train.epoch
    if n == 7:
        ckpt_dict = torch.load(
            os.path.join("./weights", f"{n}x{n}_p03_r2_l{L}_h{H}_grok{epoch}.tar")
        )
    else:
        ckpt_dict = torch.load(os.path.join("./weights", f"{n}x{n}_p03_r{r}.tar"))
    new_epoch = ckpt_dict["epoch"]
    print(f"epoch: {new_epoch} L: {L}")

    model = TransformerModel(vocab_size=ckpt_dict["vocab_size"], args=args.model).to(
        device
    )
    model.load_state_dict(ckpt_dict["model"])
    model.eval()

    attention_scores = model._backbone(
        input_ids=X_mask_test,
        attention_mask=test_att,
        output_attentions=True,
    ).attentions
        

    if mask_type != "none":
        plt.imshow(mask_test[0].view(m, n).detach().cpu().numpy(), cmap="viridis")
        plt.savefig(
            f"./att_scores_row/mask_{mask_type}{mask1}{mask2}{obs1}{obs2}_{new_epoch}.png"
        )
        plt.close()

    if H > 1:
        fig, axes = plt.subplots(
            nrows=args.model.num_hidden_layers,
            ncols=args.model.num_attention_heads,
            figsize=(40, 25),
            sharey=True,
        )
        for i in range(L):
            for j in range(H):
                im = axes[i, j].imshow(
                    np.mean(
                        attention_scores[i][:, j, :, :].detach().cpu().numpy(),
                        axis=0,
                    ),
                    cmap="viridis",
                    # origin="lower",
                )
                axes[i, j].tick_params(axis="both", labelsize=30)
                axes[i, j].xaxis.tick_top()
                axes[i, j].set_xticks([0, m * n - 1])  # , fontsize=14)
                axes[i, j].set_yticks([0, m * n - 1])  # , fontsize=14)
                if L > 4:
                    cb = fig.colorbar(
                        im, ax=axes[i, j], location="bottom", shrink=0.99, pad=0.02
                    )
                    cb.ax.tick_params(labelsize=20)
                    cb.minorticks_on()

                axes[0, j].set_title(f"Head {j + 1}", fontsize=40, pad=50)

                # Plot mask pattern below heatmap for structured mask expt

                # mask_divider = make_axes_locatable(axes[i, j])
                # mask_ax = mask_divider.append_axes("bottom", size="5%", pad=0.5)
                # mask_ax.imshow(
                #     mask_test[0, :].view(-1).detach().cpu().numpy()[np.newaxis, :],
                #     aspect="auto",
                #     cmap="viridis",
                # )
                # mask_ax.set_xticks([])
                # mask_ax.set_yticks([])

            axes[i, 0].set_ylabel(f"Layer {i + 1}", fontsize=40)
            if L == 4:
                axes[i, 0].yaxis.labelpad = 50
    else:
        fig, axes = plt.subplots(
            nrows=args.model.num_hidden_layers,
            ncols=args.model.num_attention_heads,
            figsize=(5, 35),
            # sharey=True,
        )
        for i in range(L):

            im = axes[i].imshow(
                np.mean(
                    attention_scores[i][:, 0, :, :].detach().cpu().numpy(),
                    axis=0,
                ),
                cmap="viridis",
                # origin="lower",
            )
            axes[i].tick_params(axis="both")
            axes[i].xaxis.tick_top()
            axes[i].set_xticks([0, m * n - 1])
            axes[i].set_yticks([0, m * n - 1])

    fig.tight_layout()
    fig.savefig(
        f"./att_scores_row/att_n{args.data.n}_r{args.data.rank}_l{L}_h{H}_{mask_type}{mask1}{mask2}{obs1}{obs2}_{new_epoch}.pdf"
    )

    output = model(X_mask_test, attention_mask=test_att).view(X_test.shape).cpu()
    print_output(0, 0, 0, m, n, r, output[0], X_test[0], mask_test[0])
    print(F.mse_loss(output, X_test.cpu()).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = DotMap(yaml.safe_load(f))

    main(config)

