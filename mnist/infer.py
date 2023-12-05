from __future__ import absolute_import

import git
import hydra
from data.my_data_module import MyDataModule
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from model.my_model import MyResnet
from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def test(cfg: DictConfig):
    model = MyResnet.load_from_checkpoint(cfg.test.weights_path)
    data_module = MyDataModule(
        data_dir=cfg.data.data_dir,
    )

    mlf_logger = MLFlowLogger(
        experiment_name="test_metrics",
        tracking_uri=cfg.logger.uri,
    )
    repo = git.Repo(search_parent_directories=True)
    hexsha = repo.head.object.hexsha
    mlf_logger.log_hyperparams({"git_commit_hash": hexsha})

    trainer = Trainer(
        logger=mlf_logger,
    )

    trainer.test(
        model,
        dataloaders=data_module,
    )


if __name__ == "__main__":
    test()
