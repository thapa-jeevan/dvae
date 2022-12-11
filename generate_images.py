import argparse
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from transformer import VQGANTransformer
from dvae import DiscreteVAE

parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument('--image-size', type=int, default=128, help='Image height and width.)')
parser.add_argument('--transformer-checkpoint-path', type=str, default='./checkpoints/transformer_last_ckpt.pt',
                    help='Path to checkpoint.')
parser.add_argument('--vae-checkpoint-path', type=str, default='./checkpoints/vae_last_ckpt.pt',
                    help='Path to checkpoint.')

parser.add_argument('--epoch', type=float, default=None, help='Training epoch if script called by training pipeline.')

if __name__ == '__main__':
    print("Generating Images")
    args = parser.parse_args()
    num_generations = 2

    transformer = VQGANTransformer(pkeep=1).cuda()
    transformer.load_state_dict(torch.load(args.transformer_checkpoint_path))

    vae = DiscreteVAE(
        image_size=args.image_size,
        num_layers=3,
        channels=3,
        num_tokens=2048,
        codebook_dim=256,
        hidden_dim=128,
        temperature=0
    ).cuda()
    vae.load_state_dict(torch.load(args.vae_checkpoint_path))

    print("Loaded state dict of Transformer")

    for i in tqdm(range(num_generations)):
        import numpy as np

        # with open("data/dvae_codes/1.npy", "rb") as f:
        #     array = np.load(f)
        #
        # start_indices = torch.Tensor(array[:4, :200] + 1).long().cuda()

        start_indices = torch.zeros((4, 0)).long().cuda()
        sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
        sos_tokens = sos_tokens.long().cuda()
        sample_indices = transformer.sample(start_indices, sos_tokens, temperature=1, steps=256)
        # TODO: Check sample indices

        sample_indices -= 1

        sampled_imgs = vae.decode(sample_indices)
        img_save_path = os.path.join("results", "transformer_training",
                                     (f"epoch{args.epoch}" if args.epoch else "") + f"transformer_{i}.jpg")
        save_image(sampled_imgs, img_save_path, nrow=4)