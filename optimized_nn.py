from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import get_dataloader
from net import Net

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))
    return torch.sqrt(F.mse_loss(pred, targ))/denom

def test_accuracy(net):
    testset = get_dataloader('data/', 'test')
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4
    )

    total_loss = 0
    total = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            total_loss += rmse(outputs, labels)

    return total_loss / total

def train_optimized(l1, l2, lr, batch_size, epochs=1000):    
    model_feynman = Net(l1, l2).to(device)
    trainset = get_dataloader('./data', 'train')
    valset = get_dataloader('./data', 'validation')

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=batch_size
    )

    optimizer = optim.Adam(model_feynman.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75)

    st_time = time.time()

    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_steps = 0
        running_loss = 0
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model_feynman(inputs)
            loss = rmse_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model_feynman(inputs)
                total += labels.size(0)

                loss = rmse_loss(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
        print('Validation', np.round(val_loss / val_steps, 5), optimizer.param_groups[0]['lr'])
        scheduler.step(val_loss / val_steps)

    print(time.time() - st_time, test_accuracy(model_feynman))

if __name__ == "__main__":
    train_optimized(l1=64, l2=256, lr=1e-2, batch_size=256)