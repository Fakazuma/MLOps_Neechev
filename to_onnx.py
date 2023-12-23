from __future__ import absolute_import

import hydra
from omegaconf import DictConfig
import lightning.pytorch as pl
import torch

from my_model import MyResnet


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def convert_to_onnx(cfg: DictConfig):
    model: pl.LightningModule = MyResnet.load_from_checkpoint(cfg.to_onnx.weights_path)
    input_sample = torch.randn(1, 1, 28, 28)
    model.to_onnx(
        file_path='my_resnet.onnx',
        input_sample=input_sample,
        export_params=True,
        do_constant_folding=True,
        input_names=['IMAGES'],
        output_names=['CLASS_PROBS'],
        dynamic_axes={'IMAGES': {0: 'BATCH_SIZE'}, 'CLASS_PROBS': {0: 'BATCH_SIZE'}},
    )


if __name__ == "__main__":
    convert_to_onnx()
