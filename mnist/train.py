from __future__ import absolute_import

import hydra
from data.my_data_module import MyDataModule
from lightning import Trainer
from model.my_model import MyResnet
from omegaconf import DictConfig


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
        default_root_dir=cfg.trainer.log_dir,
        max_epochs=3,
        accumulate_grad_batches=1,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    train()
