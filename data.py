import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import os

class ExampleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)

def get_dataloader(data_dir, datasplit='train'):
    data = np.loadtxt(os.path.join(data_dir, 'example_data.txt'))
    x, y = data[:, :-1], data[:, -1]
    y = y[:, np.newaxis]

    npoints = len(x)
    train_idx, val_idx = int(npoints * 0.8), int(npoints * 0.9)

    train_x, train_y = x[:train_idx], y[:train_idx]
    val_x, val_y = x[train_idx:val_idx], y[train_idx:val_idx]
    test_x, test_y = x[val_idx:], y[val_idx:]

    if datasplit == 'train':
        return ExampleDataset(train_x, train_y)
    elif datasplit == 'validation':
        return ExampleDataset(val_x, val_y)
    elif datasplit == 'test':
        return ExampleDataset(test_x, test_y)

if __name__ == "__main__":
    print(len(get_dataloader()))