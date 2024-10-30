import argparse
import yaml
from dotmap import DotMap
import wandb


import torch
import torch.nn as nn
from torch.optim import Adam

# from optim import AdamWithWarmup, AdamCosineWithWarmup

from torch.nn import MSELoss
import torch.nn.functional as F

from data import RealMatrix
from model import TransformerModel
from utils import print_output, nuc_norm_solver

torch.set_printoptions(precision=4)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device}")


def train_epoch(
    model,
    loss_fn,
    optimizer,
    epoch,
    data_sampler,
    ckpt_dir,
    best_loss,
    args,
):
    # Data ------------------------------------------------------------------
    m, n, r_t, r_e = args.data.m, args.data.n, args.data.train_rank, args.data.test_rank

    X_mask_tr, X_tr, mask_tr = data_sampler.vmap_sample(
        n_samples=args.train.num_train, m=m, n=n, r=r_t, p_mask=args.data.train_p_mask
    )
    X_mask_tr.requires_grad_(False)
    X_tr.requires_grad_(False)
    mask_tr.requires_grad_(False)

    att_mask_tr = torch.ones_like(X_mask_tr).to(device)
    att_mask_tr.requires_grad_(False)

    # Train -----------------------------------------------------------------
    train_loss = 0.0

    model.train()
    optimizer.zero_grad()

    output = model(X_mask_tr, attention_mask=att_mask_tr).view(X_tr.shape)
    loss = loss_fn(output, X_tr)

    loss.backward()
    optimizer.step()

    # Eval & Logging ---------------------------------------------------------
    with torch.no_grad():
        model.eval()
        train_output = model(X_mask_tr, attention_mask=att_mask_tr).view(X_tr.shape)
        train_loss = (
            F.mse_loss(input=train_output, target=X_tr, reduction="mean")
            .detach()
            .item()
        )

        # grokking metrics
        mask_loss = (
            F.mse_loss(
                input=train_output[mask_tr.view(train_output.shape) == 0],
                target=X_tr[mask_tr.view(train_output.shape) == 0],
                reduction="mean",
            )
            .detach()
            .item()
        )
        obs_loss = (
            F.mse_loss(
                input=train_output[mask_tr.view(train_output.shape) == 1],
                target=X_tr[mask_tr.view(train_output.shape) == 1],
                reduction="mean",
            )
            .detach()
            .item()
        )
        mean_mask_value = (
            torch.mean(torch.abs(train_output[mask_tr.view(train_output.shape) == 0]))
            .detach()
            .item()
        )

        X_mask_ev, X_ev, mask_ev = data_sampler.vmap_sample(
            n_samples=args.train.num_eval, m=m, n=n, r=r_e, p_mask=args.data.test_p_mask
        )
        X_mask_ev.requires_grad_(False)
        X_ev.requires_grad_(False)
        mask_ev.requires_grad_(False)

        att_mask_ev = torch.ones_like(X_mask_ev).to(device)
        att_mask_ev.requires_grad_(False)

        eval_output = model(X_mask_ev, attention_mask=att_mask_ev).view(X_ev.shape)
        eval_loss = (
            F.mse_loss(input=eval_output, target=X_ev, reduction="mean").detach().item()
        )

        # eval_nuc_norm_loss = nuc_norm_solver(
        #     X_ev.detach().numpy(), mask_ev.detach().numpy()
        # )

        idx = torch.randint(low=0, high=X_ev.shape[0], size=(1,)).item()
        print_output(
            epoch=epoch,
            train_loss=train_loss,
            eval_loss=eval_loss,
            # eval_nuc_norm_loss=eval_nuc_norm_loss,
            m=m,
            n=n,
            r=r_t,
            output=eval_output[idx],
            X=X_ev[idx],
            mask=mask_ev[idx],
            mask_loss=mask_loss,
            obs_loss=obs_loss,
            mean_mask_value=mean_mask_value,
        )
        del train_output, eval_output

    if args.wandb.log:
        wandb.log(
            {
                "train": train_loss,
                "eval": eval_loss,
                "mask loss": mask_loss,
                "obs loss": obs_loss,
                "mask value": mean_mask_value,
            }
        )

    if args.train.save_ckpt:
        if (epoch + 1) % args.train.save_freq == 0:
            model.train()
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "vocab_size": len(data_sampler.vocab.keys()),
                },
                f"./{ckpt_dir}.tar",
            )
            print(f"saved state at epoch {epoch} to {f'./{ckpt_dir}.tar'}")

            if args.wandb.log:
                model_wandb = wandb.Artifact(
                    f"model_{ckpt_dir}_step{epoch}", type="model"
                )
                model_wandb.add_file(f"./{ckpt_dir}.tar")
                wandb.log_artifact(model_wandb)
                print("model uploaded to wandb")

    return eval_loss


# -----------------------------------------------------------------------


def main(args, ckpt_dir):
    data_sampler = RealMatrix(args.data, device)

    model = TransformerModel(
        vocab_size=len(data_sampler.vocab.keys()), args=args.model
    ).to(device)

    mse_loss = MSELoss(reduction="mean")
    optim = Adam(model.parameters(), lr=args.train.lr)

    state_dict = model.state_dict()
    post_state_dict = torch.load(f"./weights/7x7_p03_r2_l4_h8_grok49999.tar")['model']
   
    with torch.no_grad():
        for k in list(state_dict.keys()):
            if ("attention" in k) or ("intermediate" in k) or ("output" in k) or ("read" in k):
                state_dict[k] = post_state_dict[k].clone().detach()
    
    model.load_state_dict(state_dict)
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ("attention" in name) or ("intermediate" in name) or ("output" in name) or ("read" in name):
                param.requires_grad = False
                print(f"frozen {name}")
    
    start = 0
    if args.train.restore_ckpt:
        model.train()
        ckpt = torch.load(f"{ckpt_dir}.tar")
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start = ckpt["epoch"] + 1
        vocab_size = ckpt["vocab_size"]
        print(f"loading model state at epoch {start - 1} with loss {ckpt['loss']}")

    # if args.train.save_ckpt and not os.path.isdir(ckpt_dir):
    #     os.mkdir(ckpt_dir)

    if args.wandb.log:
        wandb_run_name = ckpt_dir
        wandb.login(key="")
        wandb.init(project="tf-matcomp", name=wandb_run_name, config=args)
        wandb.watch(model)

    print(model.config)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    best_loss = 1000.0
    for epoch in range(start, start + args.train.epochs):
        eval_loss = train_epoch(
            model=model,
            loss_fn=mse_loss,
            optimizer=optim,
            epoch=epoch,
            data_sampler=data_sampler,
            ckpt_dir=ckpt_dir,
            best_loss=best_loss,
            args=args,
        )
        best_loss = min(eval_loss, best_loss)
    if args.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = DotMap(yaml.safe_load(f))

    ckpt_dir = str(args.config).split(".")[0].split("/")[1]
    print(config)
    print(ckpt_dir)
    main(config, ckpt_dir)


