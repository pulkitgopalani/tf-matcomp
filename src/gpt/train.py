from dotmap import DotMap
import yaml
import argparse

import torch
import torch.nn.functional as F
from torch.optim import Adam
import wandb

from model import GPT
from model_linear import GPTLinear
from data import *

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_printoptions(threshold=100000000)


def train_step(model, optim, data_sampler, step, config):
    dt = torch.bfloat16 if config.train.bf16 else torch.float32
    prec = config.data.prec

    n_train, n_test, num_tokens = (
        config.data.n_train,
        config.data.n_test,
        config.data.num_tokens,
    )

    L, H = config.model.n_layer, config.model.n_head

    data = data_sampler.sample(
        n_samples=n_train + n_test,
        m=config.data.m,
        n=config.data.n,
        r=config.data.r,
        p_mask=config.data.p_mask,
    )

    train_data = data[:n_train, :]
    test_data = data[n_train:, :]
    # train_matrices = matrices[:n_train, :]

    seq_len = 49
    prompt_len = 50
    gen_len = 49
    acc_start = 50

    model.train()

    optim.zero_grad(set_to_none=True)

    with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=dt):
        _, loss = model(train_data[:, :-1], targets=train_data[:, 1:])
    
    loss.backward()

    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

    optim.step()

    # log train loss, train and test acc
    model.eval()
    with torch.no_grad():
        train_loss = loss.detach().item()
        train_pred = model.generate(
            idx=train_data[:, :prompt_len],
            max_new_tokens=gen_len,
        )
        test_pred = model.generate(
            idx=test_data[:, :prompt_len],
            max_new_tokens=gen_len,
        )

        mask = test_data[:, : acc_start - 1] != 0

        train_acc = torch.mean(
            (train_pred[:, acc_start:] == train_data[:, acc_start:]).to(float)
        ).item()
        test_acc = torch.mean(
            (test_pred[:, acc_start:] == test_data[:, acc_start:]).to(float)
        ).item()

        # mask_acc = F.mse_loss((test_data[:, acc_start:][mask == 0]).to(torch.float) / 10**prec, (test_pred[:, acc_start:][mask == 0]).to(torch.float) / 10**prec).item()
        # obs_acc = F.mse_loss((test_data[:, acc_start:][mask == 1]).to(torch.float) / 10**prec, (test_pred[:, acc_start:][mask == 1]).to(torch.float) / 10**prec).item()
        
        obs_acc = torch.mean(
           (
               test_data[:, acc_start:][mask == 1]
               == test_pred[:, acc_start:][mask == 1]
           ).to(float)
        ).item()

        print(f"input: {train_data[0]} \n predicted:{train_pred[0]}")

        print(
            f"Step {step} -- Train loss: {train_loss}, Train Acc: {train_acc}, Obs MSE: {obs_acc}"
        )

        if config.train.wandb:

            log_data = {
                "train loss": train_loss,
                "train acc": train_acc,
                #"test_acc": test_acc,
                # "mask_mse": mask_acc,
                "obs_acc": obs_acc,
            }

            wandb.log(log_data)
        del train_data, test_data

        if config.train.save_ckpt:
            if (step + 1) % config.train.ckpt_freq == 0:
                model.train()
                torch.save(
                    {
                        "epoch": step,
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "train_loss": train_loss,
                        "test_acc": test_acc,
                    },
                    f"./{ckpt_dir}.tar",
                )
                print(f"saved state at epoch {step} to {f'./{ckpt_dir}.tar'}")

                if config.train.wandb:
                    model_wandb = wandb.Artifact(
                        f"model_{ckpt_dir}_step{step}", type="model"
                    )
                    model_wandb.add_file(f"./{ckpt_dir}.tar")
                    wandb.log_artifact(model_wandb)
                    print("model uploaded to wandb")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")

    config_dir = parser.parse_args().config

    with open(config_dir, "r") as f:
        config = DotMap(yaml.safe_load(f))

    ckpt_dir = str(config_dir).split(".")[0].split("/")[1]

    config.model.vocab_size = 405
    config.model.block_size = 99

    data_sampler = RealMatrix(args=config.data, device=device)

    if config.model.linear:
        model = GPTLinear(config.model).to(device)
        print("LINEAR!")
    else:
        model = GPT(config.model).to(device)
    
    optim = Adam(model.parameters(), lr=config.train.lr)

    if config.train.wandb:
        wandb_run_name = ckpt_dir
        wandb.login(key="")
        wandb.init(project="tf-emergence", name=wandb_run_name, config=config)
        wandb.watch(model)

    for step in range(config.train.num_steps):
        train_step(model, optim, data_sampler, step, config)
