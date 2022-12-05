import argparse
import os

import torch
from torchvision.utils import save_image

from dataset import get_dataloader
from dvae import DiscreteVAE


def parse_args():
    parser = argparse.ArgumentParser(description='train VAE for DALLE-pytorch')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size for training (default: 24)')
    parser.add_argument('--img_folder', type=str, default="./imagedata", help='path to imageFolder (default: ./imagedata')
    parser.add_argument('--img_size', type=int, default=256, help='image size for training (default: 256)')
    parser.add_argument('--loadVAE', type=str, default="", help='name for pretrained VAE when continuing training')
    parser.add_argument('--store_folder', type=str, default="data/dvae_codes", help='folder to dump latent codes')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_size = args.img_size  # 256
    batch_size = args.batch_size  # 24

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = DiscreteVAE(
        image_size=img_size,
        num_layers=3,
        channels=3,
        num_tokens=2048,
        codebook_dim=256,
        hidden_dim=128,
        temperature=0
    )

    if args.loadVAE != "":
        vae_dict = torch.load(args.loadVAE )
        vae.load_state_dict(vae_dict)

    vae.to(device)
    train_dataloader = get_dataloader(args.batch_size, args.img_folder, args.img_size)
    store_folder = args.store_folder
    os.makedirs(store_folder)

    for batch_idx, images in enumerate(train_dataloader):
        images = images.to(device)

        with torch.no_grad():
            codes = vae.get_codebook_indices(images)

        for img_idx, code in enumerate(codes):
            img_name = f"{batch_idx * batch_size + img_idx}.png"
            img_path = os.path.join(store_folder, img_name)
            save_image(code, img_path, normalize=True)

