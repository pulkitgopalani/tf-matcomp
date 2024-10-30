import argparse
from dotmap import DotMap
import yaml
import matplotlib.pyplot as plt
import os

import torch
import numpy as np
import torch.nn.functional as F


from model import TransformerModel
from data import RealMatrix
from utils import print_output

torch.set_printoptions(precision=4)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")


def rank_approx(X, k):
    X_approx = torch.zeros_like(X)
    u, s, v = torch.linalg.svd(X)

    for i in range(X.shape[0]):
        for r in range(k):
            X_approx[i, :, :] += s[i, r] * torch.outer(u[i, :, r], v[i, r, :])

    return X_approx


def main(args):
    m, n, r = args.data.m, args.data.n, args.data.rank
    data_sampler = RealMatrix(args.data, device=device)
    L, H = args.model.num_hidden_layers, args.model.num_attention_heads
    epoch = args.train.epoch

    if args.train.random_matrix:
        print("RANDOM INPUT!")
        X_random = -1 + 2 * torch.rand(size=(args.train.num_eval, m, n))
        mask = (torch.rand(size=X_random.shape) > args.data.p_mask)
        X_mask, X = data_sampler.tokenize(X_random, mask)
    else:
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

    replace_value = args.train.replace_value
    if replace_value > -100:
        X_mask[(mask.view(X_mask.shape)) == 0] = data_sampler.vocab[str(replace_value)]
        print(f"Replaced value: {replace_value}\n")
    else:
        replace_value = 0
    
    for epoch in [args.train.epoch]:
        if m == 10:
            ckpt_dict = torch.load(f"./weights/10x10_p03_r3_grok{epoch}.tar")
        else:
            ckpt_dict = torch.load(f"./weights/7x7_p03_r2_l{L}_h{H}_grok{epoch}.tar")

        print(f"L, H: {L}, {H}, epoch: {ckpt_dict['epoch']} ----")

        model = TransformerModel(
            vocab_size=ckpt_dict["vocab_size"], args=args.model
        ).to(device)
        model.load_state_dict(ckpt_dict["model"])
        model.eval()

        model_output = model(X_mask, attention_mask=att_mask).detach().cpu()
        print(f"m: {m}, n: {n}, r: {r}, p_mask: {args.data.p_mask}")
        print(
            f"Loss: {F.mse_loss(model_output.cpu(), X.view(model_output.shape).cpu(), reduction='mean').item()}"
        )
        print(f"Mask replace loss: {torch.mean((model_output[mask==0] - replace_value)**2)}")
        print(
            f"L_obs: {F.mse_loss(model_output.cpu()[mask==1], X.view(model_output.shape).cpu()[mask==1], reduction='mean').item()}"
        )

        print_output(0, 0, 0, m, n, r, output=model_output[0], X=X[0], mask=mask[0])
        del model, model_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = DotMap(yaml.safe_load(f))

    main(config)
