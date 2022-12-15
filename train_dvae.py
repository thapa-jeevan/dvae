import argparse

import torch
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image

from dataset_dvae import get_dataloader
from model_dvae import DiscreteVAE


def parse_args():
    parser = argparse.ArgumentParser(description='train VAE for DALLE-pytorch')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--img_folder', type=str, default="data/CelebAMask-HQ/CelebA-HQ-img",
                        help='path to image folder')
    parser.add_argument('--img_size', type=int, default=128, help='image size for training')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--tempsched', action='store_true', default=False, help='use temperature scheduling')
    parser.add_argument('--temperature', type=float, default=0.9, help='vae temperature (default: 0.9)')
    parser.add_argument('--name', type=str, default="vae", help='experiment name')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='name for pretrained vae to continue training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    log_interval = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = DiscreteVAE(
        image_size=args.img_size,
        num_tokens=2048,
        codebook_dim=256,
        num_layers=3,
        hidden_dim=128,
        channels=3,
        temperature=args.temperature
    ).to(device)

    if args.pretrained_checkpoint:
        vae.load_state_dict(torch.load(args.pretrained_checkpoint))

    train_dataloader = get_dataloader(args.batch_size, args.img_size)

    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    if args.tempsched:
        vae.temperature = args.temperature
        dk = 0.7 ** (1 / len(train_dataloader))
        print('Scale Factor:', dk)

    for epoch in range(args.n_epochs):

        train_loss = 0
        for batch_idx, images in enumerate(train_dataloader):
            images = images.to(device)
            recons = vae(images)
            loss = F.smooth_l1_loss(images, recons) + F.mse_loss(images, recons)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    epoch, batch_idx * len(images), len(train_dataloader.dataset),
                           100. * batch_idx / len(train_dataloader),
                           loss.item() / len(images)))

        if args.tempsched:
            vae.temperature *= dk
            print("Current temperature: ", vae.temperature)

        k = 8
        with torch.no_grad():
            codes = vae.get_codebook_indices(images)
            imgx = vae.decode(codes)
        grid = torch.cat([images[:k], recons[:k], imgx[:k]])
        save_image(grid,
                   'results/' + args.name + '_epoch_' + str(epoch) + '.png', normalize=True)

        print('====> Epoch: {} Average loss: {:.8f}'.format(
            epoch, train_loss / len(train_dataloader.dataset)))

        torch.save(vae.state_dict(), "./checkpoints/" + args.name + "-" + str(epoch) + ".pth")
