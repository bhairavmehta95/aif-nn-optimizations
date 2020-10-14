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

device = torch.device("cpu")
torch.set_default_dtype(torch.float64)


def train_regression(config, checkpoint_dir=None, data_dir=None):
    net = Net(config["l1"], config["l2"])
    net.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])

    # if checkpoint_dir:
    #     model_state, optimizer_state = torch.load(
    #         os.path.join(checkpoint_dir, "checkpoint"))
    #     net.load_state_dict(model_state)
    #     optimizer.load_state_dict(optimizer_state)

    trainset = get_dataloader(data_dir, 'train')
    valset = get_dataloader(data_dir, 'validation')

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(config["batch_size"])
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=int(config["batch_size"])
    )
  

    for epoch in range(50):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = torch.sqrt(criterion(outputs, labels))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                loss = torch.sqrt(criterion(outputs, labels))
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps))

    print("Finished Training")


def test_accuracy(net, device="cpu"):
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
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            total_loss += torch.sqrt(criterion(outputs, labels))

    return total_loss / total


def main(num_samples=10, max_num_epochs=50, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    checkpoint_dir = os.path.abspath("./checkpoints")
    config = {
        "l1": tune.choice([32, 64, 128, 256]),
        "l2": tune.choice([32, 64, 128, 256]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([64, 128, 256])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    st_time = time.time()
    result = tune.run(
        partial(train_regression, checkpoint_dir=checkpoint_dir, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    print("Total Time", st_time - time.time())


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=20, max_num_epochs=50, gpus_per_trial=0)