import lightning.pytorch as pl
import transformers
from pytorch_lightning.utilities import grads
from torch import nn, optim


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
        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.fc = nn.Linear(in_features=1000, out_features=10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv(x)
        x = self.model(x).logits
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(grads.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
