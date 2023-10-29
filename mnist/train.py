from __future__ import absolute_import

import hydra
from lightning import Trainer
from omegaconf import DictConfig

from data.my_data_module import MyDataModule
from model.my_model import MyResnet


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    data_module = MyDataModule(
        data_dir=cfg.data.data_dir,
        val_part=cfg.data.val_part,
        batch_size=cfg.data.batch_size,
        dataloader_num_workers=cfg.data.dataloader_num_workers,
    )

    model = MyResnet(
        model_name=cfg.model,
        learning_rate=cfg.optimizer.lr,
    )

    trainer = Trainer(
        default_root_dir=cfg.trainer.root_log_dir,
        max_epochs=cfg.trainer.max_epochs,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
