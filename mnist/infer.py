from __future__ import absolute_import

import hydra
from omegaconf import DictConfig
from lightning import Trainer

from model.my_model import MyResnet
from data.my_data_module import MyDataModule


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def test(cfg: DictConfig):
    model = MyResnet.load_from_checkpoint(cfg.test.weights_path)
    data_module = MyDataModule(data_dir='./data')
    trainer = Trainer(devices=1)
    trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    test()
