import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from dataset_dvae import get_dataloader
from model_dvae import DiscreteVAE


def parse_args():
    parser = argparse.ArgumentParser(description='train dVAE')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--img_folder', type=str, default="data/CelebAMask-HQ/CelebA-HQ-imgQ",
                        help='path to image folder')
    parser.add_argument('--img_size', type=int, default=256, help='image size for training')
    parser.add_argument('--vae_checkpoint_path', type=str, default=None,
                        help='path to pretrained vae when continuing training')
    parser.add_argument('--store_folder', type=str, default="data/dvae_codes", help='folder to dump latent codes')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.store_folder)

    vae = DiscreteVAE(
        image_size=args.img_size,
        num_layers=3,
        channels=3,
        num_tokens=2048,
        codebook_dim=256,
        hidden_dim=128,
        temperature=0
    ).to(device)

    if args.vae_checkpoint_path:
        vae.load_state_dict(torch.load(args.vae_checkpoint_path))

    train_dataloader = get_dataloader(args.batch_size, args.img_folder, args.img_size)
    store_folder = args.store_folder

    for batch_idx, images in tqdm(enumerate(train_dataloader)):
        images = images.to(device)

        with torch.no_grad():
            codes = vae.get_codebook_indices(images)
        save_path = os.path.join(store_folder, f"{batch_idx}.npy")
        with open(save_path, "wb") as f:
            np.save(f, codes.detach().cpu().numpy())
