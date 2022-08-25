from typing import Dict, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models.MainModel import EncoderModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PLEncoder(LightningModule):
    def __init__(
        self,
        config: Dict,
        optimizer_config: Dict,
        class_weight: Union[int, float] = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = EncoderModel(**config)
        self.optimizer_config = optimizer_config
        self.class_weight = torch.tensor(class_weight)

    def forward(self, x, src_mask):
        return self.model(x, src_mask)

    def loss_function(self, output, label):
        loss = F.mse_loss(output, label)
        return loss

    def training_step(self, batch, batch_idx):
        x = batch["input_ids"]
        src_mask = batch["attention_mask"]

        y = batch["labels"].to(self.device).float()
        y_hat = torch.argmax(self(x, src_mask), dim=1).float()
        y_hat.requires_grad = True

        loss = self.loss_function(y_hat, y)

        log_dict = {
            "train/loss": loss,
        }
        self.log_dict(log_dict, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["input_ids"]
        src_mask = batch["attention_mask"]

        y = batch["labels"].to(self.device).float()
        y_hat = torch.argmax(self(x, src_mask), dim=1).float()
        y_hat.requires_grad = True

        loss = self.loss_function(y_hat, y)

        log_dict = {
            "eval/loss": loss,
        }
        self.log_dict(log_dict, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.optimizer_config["lr"],
        )
        return optimizer
