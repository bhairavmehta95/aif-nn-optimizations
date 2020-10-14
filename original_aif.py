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

from data import get_dataloader
from net import Net

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

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
            total += labels.size(0)

            total_loss += rmse_loss(outputs, labels)

    return total_loss / total

def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))
    return torch.sqrt(F.mse_loss(pred, targ))/denom

def original_AIF(epochs=1000, lrs=1e-3, N_red_lr=4):
    model_feynman = Net(128, 64).to(device)
    trainset = get_dataloader('./data', 'train')
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=256
    )

    check_es_loss = 1000
    epoch_counter = 0

    st_time = time.time()

    for i_i in range(N_red_lr):
        optimizer_feynman = optim.Adam(model_feynman.parameters(), lr = lrs)
        for epoch in range(epochs):
            for i, data in enumerate(trainloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer_feynman.zero_grad()
                
                loss = rmse_loss(model_feynman(inputs),labels)
                loss.backward()
                optimizer_feynman.step()

            # Early stopping
            if epoch%20==0 and epoch>0:
                if check_es_loss < loss:
                    break
                else:
                    check_es_loss = loss
            if epoch==0:
                if check_es_loss < loss:
                    check_es_loss = loss

            epoch_counter += 1
            if epoch_counter % 10 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch_counter, loss.item()))
    
        print(loss)
        lrs = lrs/10

    print(time.time() - st_time, test_accuracy(model_feynman))

if __name__ == "__main__":
    original_AIF()