from __future__ import absolute_import

import json
import os
from math import ceil
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from model.resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from model.utils import (
    MEAN_NORMALIZATION,
    STD_NORMALIZATION,
    count_layers,
    count_trainable_parameters,
    select_device,
)
from timm.utils.metrics import AverageMeter, accuracy
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torchvision.transforms import v2

MODELS = {
    "resnet20": {
        "a": resnet20(zero_padding=False, pretrained=False),
        "b": resnet20(zero_padding=True, pretrained=False),
    },
    "resnet32": {
        "a": resnet32(zero_padding=False, pretrained=False),
        "b": resnet32(zero_padding=True, pretrained=False),
    },
    "resnet44": {
        "a": resnet44(zero_padding=False, pretrained=False),
        "b": resnet44(zero_padding=True, pretrained=False),
    },
    "resnet56": {
        "a": resnet56(zero_padding=False, pretrained=False),
        "b": resnet56(zero_padding=True, pretrained=False),
    },
    "resnet101": {
        "a": resnet110(zero_padding=False, pretrained=False),
        "b": resnet110(zero_padding=True, pretrained=False),
    },
}


def train_mnist(arch: str, zero_padding: bool) -> None:
    option = "b" if zero_padding else "a"
    model = MODELS[arch][option]

    device = torch.device(select_device())

    model.to(device)

    # Count the total number of trainable parameters
    trainable_parameters = count_trainable_parameters(model=model)
    print(f"# of Trainable Parameters: {trainable_parameters}")

    # Count the total number of layers
    total_layers = count_layers(model=model)
    print(f"# of Layers: {total_layers}")

    # Initialize/Define train transformation
    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=(32, 32), padding=4),
            T.ToTensor(),
            v2.Lambda(lambda x: x.repeat(3, 1, 1)),  # x.unsqueeze(1).repeat(3, 1, 1)
            T.Normalize(mean=MEAN_NORMALIZATION, std=STD_NORMALIZATION),
        ]
    )

    # Initialize/Define test transformation
    test_transform = T.Compose(
        [
            T.ToTensor(),
            v2.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean=MEAN_NORMALIZATION, std=STD_NORMALIZATION),
        ]
    )

    # Define the batch size before preparing the dataloaders
    BATCH_SIZE = 64

    train_dataset = MNIST(
        root="data", train=True, download=True, transform=train_transform
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )

    test_dataset = MNIST(
        root="data", train=False, download=True, transform=test_transform
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    # Define training parameters as described in the original paper
    ITERATIONS = 10
    EPOCHS = ceil(ITERATIONS / len(train_dataloader))
    LR_MILESTONES = [
        ceil(32000 / len(train_dataloader)),
        ceil(48000 / len(train_dataloader)),
    ]

    criterion = (
        nn.CrossEntropyLoss().cuda()
        if torch.cuda.is_available()
        else nn.CrossEntropyLoss()
    )
    optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=9e-1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_MILESTONES, gamma=1e-1
    )

    best_prec1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Initialize AverageMeters before training
        train_losses = AverageMeter()
        train_prec1 = AverageMeter()

        model.train()
        start_time = time()

        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs, loss = outputs.float(), loss.float()

            prec1 = accuracy(outputs.data, labels)[0]
            train_losses.update(loss.item(), inputs.size(0))
            train_prec1.update(prec1.item(), inputs.size(0))

        train_loss = train_losses.avg
        train_acc = train_prec1.avg / 100
        train_error = 1.0 - train_acc
        train_time = time() - start_time

        # wandb.log({
        #     'train_loss': train_loss, 'train_acc': train_acc,
        #     'train_error': train_error, 'train_time': train_time
        # }, step=epoch)

        scheduler.step()

        # Initialize AverageMeters before evaluation
        test_losses = AverageMeter()
        test_prec1 = AverageMeter()

        model.eval()
        start_time = time()

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                outputs, loss = outputs.float(), loss.float()

                prec1 = accuracy(outputs.data, labels)[0]
                test_losses.update(loss.item(), inputs.size(0))
                test_prec1.update(prec1.item(), inputs.size(0))

            test_loss = test_losses.avg
            test_acc = test_prec1.avg / 100
            test_error = 1.0 - test_acc
            test_time = time() - start_time

            # wandb.log({
            #     'test_loss': test_loss, 'test_acc': test_acc,
            #     'test_error': test_error, 'test_time': test_time
            # }, step=epoch)

        if best_prec1 is None:
            best_prec1 = test_acc
        if best_prec1 <= test_acc:
            torch.save(
                model.state_dict(),
                os.path.join("weights/", f"{arch}{option}-mnist.pth"),
            )
            with open(os.path.join("logs/", f"{arch}{option}-mnist.json"), "w") as f:
                json.dump(
                    {"epoch": epoch, "train_prec1": train_acc, "test_prec1": test_acc},
                    f,
                )
            best_prec1 = test_acc


if __name__ == "__main__":
    train_mnist(arch="resnet20", zero_padding=True)
