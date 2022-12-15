import glob
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class ImageFolder(Dataset):
    def __init__(self, img_path_list, transform):
        self.img_path_list = img_path_list
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, item):
        img = Image.open(self.img_path_list[item])
        return self.transform(img)


def get_dataloader(batch_size, img_size):
    img_path_list_celeba = glob.glob(os.path.join("data", "celeba", "*"))
    img_path_list_ffhq = glob.glob(os.path.join("data", "ffhq", "*", "*"))
    img_path_list = img_path_list_celeba + img_path_list_ffhq

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = ImageFolder(img_path_list, transform=transform)

    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=batch_size, shuffle=True)
    return train_loader
