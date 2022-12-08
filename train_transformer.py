import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import VQGANTransformer

from dataset_transformer import get_dataloader


def configure_optimizers(model, learning_rate=4.5e-06):
    decay, no_decay = set(), set()
    whitelist_weight_modules = (nn.Linear, )
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.transformer.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn

            if pn.endswith("bias"):
                no_decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)

            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    no_decay.add("pos_emb")

    param_dict = {pn: p for pn, p in model.transformer.named_parameters()}

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
    return optimizer


def train(train_dataloader, model, optimizer, epochs, experiment_name):
    for epoch in range(epochs):
        train_loss = 0
        with tqdm(range(len(train_dataloader))) as pbar:
            for i, (codes,) in zip(pbar, train_dataloader):
                optimizer.zero_grad()
                codes = codes.long().cuda()
                logits, targets = model(codes)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                pbar.update(0)
                train_loss += loss.cpu().detach().numpy().item()

        print(f"\nTraining Loss: {train_loss / len(train_dataloader)}\n")
        checkpoint_path = os.path.join("checkpoints", f"{experiment_name}_transformer_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--array-folder', type=str, help='Folder containing dvae codes.')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Original checkpoint path to finetune.')

    parser.add_argument('--experiment-name', type=str, default=None, help='Name for the experiment.')

    parser.add_argument('--batch-size', type=int, default=32, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=3e-04, help='Learning rate.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()

    model = VQGANTransformer(args.sos_token, args.pkeep)
    model.cuda()

    if args.checkpoint_path:
        model.load_state_dict(torch.load(args.checkpoint_path))

    optimizer = configure_optimizers(model, args.learning_rate)

    train_dataloader = get_dataloader(args.batch_size, args.array_folder)
    train(train_dataloader, model, optimizer, args.epochs, args.experiment_name)
