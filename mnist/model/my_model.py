from collections import defaultdict
from typing import Tuple

import lightning.pytorch as pl
import torch
import transformers
from pytorch_lightning.utilities import grads
from torch import nn, optim
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
)


class MyResnet(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "microsoft/resnet-50",
        learning_rate: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = transformers.ResNetForImageClassification.from_pretrained(
            model_name,
            return_dict=True,
        )
        self.learning_rate = learning_rate
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, bias=False)
        self.fc = nn.Linear(in_features=1000, out_features=10, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv(x)
        x = self.model(x).logits
        x = self.fc(x)
        return x

    def training_step(self, batch):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_start(self) -> None:
        self.test_preds_on_batch = defaultdict(list)

    def test_step(self, batch: Tuple[torch.Tensor, ...]) -> None:
        x, y = batch
        output = self(x)
        self.test_preds_on_batch["preds"].append(output)
        self.test_preds_on_batch["true"].append(y)

    def on_test_epoch_end(self) -> None:
        preds = torch.cat(self.test_preds_on_batch["preds"], dim=0)
        true = torch.cat(self.test_preds_on_batch["true"], dim=0)
        accuracy = MulticlassAccuracy(num_classes=10).to(self.device)
        f1 = MulticlassF1Score(num_classes=10).to(self.device)
        auc = MulticlassAUROC(num_classes=10).to(self.device)
        self.log("Accuracy", accuracy(preds, true))
        self.log("F1-Score", f1(preds, true))
        self.log("ROC-AUC", auc(preds, true))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        self.log(
            "grad_2.0_norm_total",
            grads.grad_norm(self, norm_type=2)["grad_2.0_norm_total"],
        )
        super().on_before_optimizer_step(optimizer)
