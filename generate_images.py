import argparse
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from transformer import VQGANTransformer
from dvae import DiscreteVAE

parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
parser.add_argument('--image-size', type=int, default=128, help='Image height and width.)')
parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
parser.add_argument('--transformer-checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt',
                    help='Path to checkpoint.')
parser.add_argument('--vae-checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt',
                    help='Path to checkpoint.')
parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
parser.add_argument('--batch-sizecheck', type=int, default=20, help='Input batch size for training.')
parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')
parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')

if __name__ == '__main__':
    args = parser.parse_args()
    num_generations = 100

    transformer = VQGANTransformer(args).cuda()
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
        start_indices = torch.zeros((4, 0)).long().cuda()
        sos_tokens = torch.ones(start_indices.shape[0], 1) * 0
        sos_tokens = sos_tokens.long().cuda()
        sample_indices = transformer.sample(start_indices, sos_tokens, steps=256)
        # TODO: Check sample indices
        sample_indices -= 1

        sampled_imgs = vae.decode(sample_indices)
        save_image(sampled_imgs, os.path.join("results", "transformer", f"transformer_{i}.jpg"), nrow=4)
