import argparse
from dotmap import DotMap
import yaml
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


from model import TransformerModel
from data import RealMatrix
from utils import print_output

torch.set_printoptions(precision=4, threshold=100000)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")
torch.set_grad_enabled(False)


def main(args):
    m, n, r, eta = args.data.m, args.data.n, args.data.rank, args.data.eta
    L, H = args.model.num_hidden_layers, args.model.num_attention_heads

    data_sampler = RealMatrix(args.data, device=device)
    X_mask_clean, X_clean, X_mask_corr, X_corr, mask = data_sampler.sample_patch(
        n_samples=args.train.num_eval,
        m=m,
        n=n,
        r=r,
        p_mask=args.data.p_mask,
    )

    X_mask_clean = X_mask_clean.detach().requires_grad_(False)
    X_clean = X_clean.detach().requires_grad_(False)
    X_mask_corr = X_mask_corr.detach().requires_grad_(False)
    X_corr = X_corr.detach().requires_grad_(False)
    mask = mask.detach().cpu().view(-1, m * n)

    att_mask = torch.ones_like(X_mask_clean).to(device)
    att_mask.requires_grad_(False)

    epoch = args.train.epoch

    model_pth = f"./weights/7x7_p03_r2_l{L}_h{H}_grok{epoch}.tar"
    ckpt_dict = torch.load(model_pth)
    print(f"L, H: {L}, {H}, epoch: {ckpt_dict['epoch']}\n")

    mask_flag = args.train.mask_flag
    corr_mask_loss = []
    corr_obs_loss = []
    clean_mask_loss = []
    clean_obs_loss = []

    for l in range(0, 1):
        patch_mask = torch.zeros(size=(L, args.train.num_eval, H, m * n))

        mask = mask.view(-1, m * n)
        patch_mask = patch_mask.permute(0, 2, 1, 3).contiguous()
        patch_mask[3, :, :, :] = 1
        patch_mask = patch_mask.permute(0, 2, 1, 3).contiguous()
        
        extract_activation = patch_mask
        clean_states = torch.empty(
            size=(L, args.train.num_eval, H, m * n, 768 // H)
        ).to(device)

        # Extract hidden states for clean input ------------
        model = TransformerModel(
            vocab_size=ckpt_dict["vocab_size"], args=args.model
        ).to(device)
        model.load_state_dict(ckpt_dict["model"])
        model.eval()

        _ = model(
            X=X_mask_clean,
            patch_states=None,
            extract_activation=extract_activation,
            attention_mask=att_mask,
        ).view(X_clean.shape)

        for l in range(L):
            clean_states[l] = (
                model._backbone.encoder.layer[l]
                .attention.self.extracted_acts.clone()
                .detach()
            )
        del model

        # Patch model with corrupted input ------------
        model = TransformerModel(
            vocab_size=ckpt_dict["vocab_size"], args=args.model
        ).to(device)
        model.load_state_dict(ckpt_dict["model"])
        model.eval()

        patched_output = model(
            X=X_mask_corr,
            patch_states=(patch_mask, clean_states),
            extract_activation=None,
            attention_mask=att_mask,
        ).view(X_corr.shape)
        del model

        mask = mask.view(X_clean.shape)

        mask_patch_corr_dist = F.mse_loss(
            patched_output[mask == 0], X_corr[mask == 0], reduction="mean"
        ).item()
        corr_mask_loss.append(mask_patch_corr_dist)

        obs_patch_corr_dist = F.mse_loss(
            patched_output[mask == 1], X_corr[mask == 1], reduction="mean"
        ).item()
        corr_obs_loss.append(obs_patch_corr_dist)

        mask_patch_clean_dist = F.mse_loss(
            patched_output[mask == 0], X_clean[mask == 0], reduction="mean"
        ).item()
        clean_mask_loss.append(mask_patch_clean_dist)

        obs_patch_clean_dist = F.mse_loss(
            patched_output[mask == 1], X_clean[mask == 1], reduction="mean"
        ).item()
        clean_obs_loss.append(obs_patch_clean_dist)

        print(f"|TF(-X, patch(X)) - (-X)| at mask: {round(mask_patch_corr_dist, 4)}")
        print(f"|TF(-X, patch(X)) - (-X)| at  obs: {round(obs_patch_corr_dist, 4)}")
        print_output(0, 0, 0, m, n, r, patched_output[0], X_corr[0], mask[0])

        print(f"|TF(-X, patch(X)) - X| at mask: {round(mask_patch_clean_dist, 4)}")
        print(f"|TF(-X, patch(X)) - X| at obs: {round(obs_patch_clean_dist, 4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = DotMap(yaml.safe_load(f))

    main(config)
