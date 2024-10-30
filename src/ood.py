import argparse
from dotmap import DotMap
import yaml

import torch
import torch.nn.functional as F


from model import TransformerModel
from data import *
from utils import print_output

torch.set_printoptions(precision=4)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")


def main(args):
    m, n, r = args.data.m, args.data.n, args.data.rank
    L, H = args.model.num_hidden_layers, args.model.num_attention_heads

    if args.data.type == "uniform":
        data_sampler = RealMatrix(args.data, device=device)

    elif args.data.type == "integer":
        data_sampler = IntegerMatrix(args.data, device=device)

    elif args.data.type == "gaussian":
        data_sampler = GaussianMatrix(args.data, device=device) 

    elif args.data.type == "laplace":
        data_sampler = LaplaceMatrix(args.data, device=device)

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

    epoch = args.train.epoch

    ckpt_dict = torch.load(f"./weights/7x7_p03_r2_l4_h8_grok{epoch}.tar")

    print(f"epoch: {ckpt_dict['epoch']}\n")
    print(f"L, H: {L}, {H}\n")

    model = TransformerModel(vocab_size=ckpt_dict["vocab_size"], args=args.model).to(
        device
    )
    model.load_state_dict(ckpt_dict["model"])
    model.eval()
    

    model_output = model(X_mask, attention_mask=att_mask).detach().cpu()
    print(
        f"Total: {F.mse_loss(model_output.cpu(), X.view(model_output.shape).cpu(), reduction='mean').item()}"
    )

    print(
        f"Observed: {F.mse_loss(model_output.cpu()[mask==1], X.view(model_output.shape).cpu()[mask==1], reduction='mean').item()}"
    )
    print(
        f"Masked: {F.mse_loss(model_output.cpu()[mask==0], X.view(model_output.shape).cpu()[mask==0], reduction='mean').item()}"
    )
    print_output(0, 0, 0, m, n, r, output=model_output[0], X=X[0], mask=mask[0])
    del model_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = DotMap(yaml.safe_load(f))

    main(config)
