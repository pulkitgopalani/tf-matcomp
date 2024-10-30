import argparse
from functools import reduce
from multiprocessing import reduction
from unittest.mock import patch

from networkx import attracting_components
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
import wandb
from utils import print_output

torch.set_printoptions(precision=4, threshold=100000)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")
torch.set_grad_enabled(False)


def main(args):
    m, n, r, eta = args.data.m, args.data.n, args.data.rank, args.data.eta
    L, H = args.model.num_hidden_layers, args.model.num_attention_heads
    # patch_layer = args.train.patch_layer

    data_sampler = RealMatrix(args.data, device=device)

    X_mask, X, mask = data_sampler.sample(
        n_samples=args.train.num_eval,
        m=m,
        n=n,
        r=r,
        p_mask=args.data.p_mask,
    )

    X_mask = X_mask.detach().requires_grad_(False)
    X = X.detach().requires_grad_(False)
    mask = mask.detach().cpu()

    att_mask = torch.ones_like(X_mask).to(device).requires_grad_(False)

    pre_epoch, post_epoch = args.train.pre_epoch, args.train.post_epoch
    if args.train.fuse_dir == "pre-to-post":
        src_model_pth = f"./weights/7x7_p03_r2_l{L}_h{H}_grok{pre_epoch}.tar"
        dst_model_pth = f"./weights/7x7_p03_r2_l{L}_h{H}_grok{post_epoch}.tar"
    else:
        src_model_pth = f"./weights/7x7_p03_r2_l{L}_h{H}_grok{post_epoch}.tar"
        dst_model_pth = f"./weights/7x7_p03_r2_l{L}_h{H}_grok{pre_epoch}.tar"

    src_dict = torch.load(src_model_pth)
    dst_dict = torch.load(dst_model_pth)

    print(
        f"L, H: {L}, {H}, src epoch: {src_dict['epoch']}, dst epoch: {dst_dict['epoch']}\n"
    )

    # if args.train.fuse != 'none':
    print(f"fusing {args.train.fuse} {args.train.fuse_dir}")
    for k in dst_dict["model"].keys():
        if args.train.fuse == "att":
            # if ('embedding' in k) or ('read_out' in k):
            #    dst_dict['model'][k] = src_dict['model'][k]
            if ("key" in k) or ("query" in k) or ("value" in k):
                dst_dict["model"][k] = src_dict["model"][k].clone().detach()

        elif args.train.fuse == "mlp":
            # if 'embedding' in k or 'read_out' in k:
            #    dst_dict['model'][k] = src_dict['model'][k]
            if (
                ("attention" not in k)
                #and ("read_out" not in k)
                #and ("embedding" not in k)
            ):
                dst_dict["model"][k] = src_dict["model"][k].clone().detach()

    model = TransformerModel(vocab_size=dst_dict["vocab_size"], args=args.model).to(
        device
    )
    model.load_state_dict(dst_dict["model"])
    model.eval()

    output = model(
        X=X_mask,
        attention_mask=att_mask,
    ).view(X.shape)

    print(f"all: {F.mse_loss(output, X, reduction='mean').item()}")
    print(
        f"observed: {F.mse_loss(output[mask == 1], X[mask == 1], reduction='mean').item()}"
    )
    print(
        f"masked: {F.mse_loss(output[mask == 0], X[mask == 0], reduction='mean').item()}"
    )

    print_output(0, 0, 0, m, n, r, output[0], X[0], mask[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = DotMap(yaml.safe_load(f))

    main(config)
