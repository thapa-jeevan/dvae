import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_dataloader(batch_size, array_folder):
    code_path_list = glob.glob(os.path.join(array_folder, "*"))
    code_list = []
    for file_path in code_path_list:
        with open(file_path, "rb") as f:
            code_ = np.load(f)
            code_list.append(code_)

    codes_array = np.vstack(code_list)
    train_dataset = TensorDataset(torch.Tensor(codes_array))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader
