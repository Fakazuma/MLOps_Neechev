from __future__ import absolute_import

import git
import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from model.my_model import MyResnet
from data.my_data_module import MyDataModule


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

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        tracking_uri=cfg.logger.uri,
    )
    repo = git.Repo(search_parent_directories=True)
    hexsha = repo.head.object.hexsha
    mlf_logger.log_hyperparams({"git_commit_hash": hexsha})

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.savedir,
        filename="mnist-checkpoint",
    )

    trainer = Trainer(
        default_root_dir=cfg.trainer.root_log_dir,
        max_epochs=cfg.trainer.max_epochs,
        logger=mlf_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    train()
