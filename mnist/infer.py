from __future__ import absolute_import

from time import time

import torch
import torch.nn as nn
from timm.utils.metrics import AverageMeter, accuracy
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torchvision.transforms import v2

from .model.resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from .model.utils import MEAN_NORMALIZATION, STD_NORMALIZATION, select_device

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


def infer_mnist(arch: str, zero_padding: bool, weights: str) -> None:
    option = "b" if zero_padding else "a"
    model = MODELS[arch][option]

    model.load_state_dict(torch.load(weights))

    device = torch.device(select_device())
    model.to(device)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    test_transform = T.Compose(
        [
            T.ToTensor(),
            v2.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize(mean=MEAN_NORMALIZATION, std=STD_NORMALIZATION),
        ]
    )

    BATCH_SIZE = 64

    test_dataset = MNIST(
        root="data", train=False, download=True, transform=test_transform
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
    )

    criterion = (
        nn.CrossEntropyLoss().cuda()
        if torch.cuda.is_available()
        else nn.CrossEntropyLoss()
    )

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

    print(
        f"Test Loss:\t{test_loss}",
        f"Test Accuracy:\t{test_acc}",
        f"Test Error:\t{test_error}",
        f"Test Time:\t{test_time}",
        sep="\n",
    )


if __name__ == "__main__":
    infer_mnist(
        arch="resnet20", zero_padding=True, weights="weights/resnet20b-mnist.pth"
    )
