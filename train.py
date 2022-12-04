import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision.utils import save_image

from dataset import get_dataloader
from dvae import DiscreteVAE


def parse_args():
    parser = argparse.ArgumentParser(description='train VAE for DALLE-pytorch')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size for training (default: 24)')
    parser.add_argument('--img_folder', type=str, default="./imagedata", help='path to imageFolder (default: ./imagedata')
    parser.add_argument('--img_size', type=int, default=256, help='image size for training (default: 256)')
    parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--tempsched', action='store_true', default=False, help='use temperature scheduling')
    parser.add_argument('--temperature', type=float, default=0.9, help='vae temperature (default: 0.9)')
    parser.add_argument('--name', type=str, default="vae", help='experiment name')
    parser.add_argument('--loadVAE', type=str, default="", help='name for pretrained VAE when continuing training')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch numbering for continuing training (default: 0)')
    parser.add_argument('--clip', type=float, default=0, help='clip weights, 0 = no clipping (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_size = args.img_size  # 256
    batch_size = args.batch_size  # 24
    n_epochs = args.n_epochs  # 500
    log_interval = 10
    lr = args.lr  # 1e-4
    temperature_scheduling = args.tempsched  # True

    name = args.name  # "v2vae256"

    # for continuing training
    # set loadfn: path to pretrained model
    # start_epoch: start epoch numbering from this
    loadfn = args.loadVAE  # ""
    start_epoch = args.start_epoch  # 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = DiscreteVAE(
        image_size=img_size,
        num_layers=3,
        channels=3,
        num_tokens=2048,
        codebook_dim=256,
        hidden_dim=128,
        temperature=args.temperature
    )

    if loadfn != "":
        vae_dict = torch.load(loadfn)
        vae.load_state_dict(vae_dict)

    vae.to(device)
    train_dataloader = get_dataloader(args.batch_size, args.img_folder, args.img_size)

    optimizer = optim.Adam(vae.parameters(), lr=lr)


    def clamp_weights(m):
        if type(m) != nn.BatchNorm2d and type(m) != nn.Sequential:
            for p in m.parameters():
                p.data.clamp_(-args.clip, args.clip)


    if temperature_scheduling:
        vae.temperature = args.temperature
        dk = 0.7 ** (1 / len(train_dataloader))
        print('Scale Factor:', dk)

    for epoch in range(start_epoch, start_epoch + n_epochs):

        train_loss = 0
        for batch_idx, (images, _) in enumerate(train_dataloader):
            images = images.to(device)
            recons = vae(images)
            loss = F.smooth_l1_loss(images, recons) + F.mse_loss(images, recons)

            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if args.clip > 0:
                vae.apply(clamp_weights)

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                    epoch, batch_idx * len(images), len(train_dataloader.dataset),
                           100. * batch_idx / len(train_dataloader),
                           loss.item() / len(images)))

        if temperature_scheduling:
            vae.temperature *= dk
            print("Current temperature: ", vae.temperature)

        k = 8
        with torch.no_grad():
            codes = vae.get_codebook_indices(images)
            imgx = vae.decode(codes)
        grid = torch.cat([images[:k], recons[:k], imgx[:k]])
        save_image(grid,
                   'results/' + name + '_epoch_' + str(epoch) + '.png', normalize=True)

        print('====> Epoch: {} Average loss: {:.8f}'.format(
            epoch, train_loss / len(train_dataloader.dataset)))

        torch.save(vae.state_dict(), "./models/" + name + "-" + str(epoch) + ".pth")
