import fire

from mnist.infer import infer_mnist
from mnist.train import train_mnist


def train():
    train_mnist(arch="resnet20", zero_padding=True)


def infer():
    infer_mnist(
        arch="resnet20", zero_padding=True, weights="weights/resnet20b-mnist.pth"
    )


if __name__ == "__main__":
    fire.Fire()
