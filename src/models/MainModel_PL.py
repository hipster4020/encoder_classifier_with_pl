from typing import Dict, Union

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models.MainModel import EncoderModel


class PLEncoder(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        class_weight: Union[int, float] = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = EncoderModel(**model_config)
        self.optimizer_config = optimizer_config
        self.class_weight = torch.tensor(class_weight)

    def forward(self, input_ids, spectrogram):
        return self.model(input_ids, spectrogram)

    def loss_function(self, output, label):
        loss = F.mse_loss(output, label)
        return loss

    def training_step(self, batch, batch_idx):
        input_ids, spectrogram = batch

        output = self(input_ids, spectrogram)
        loss = self.loss_function(output)

        log_dict = {
            "train/loss": loss,
        }

        self.log_dict(log_dict, on_epoch=True)
        return log_dict

    def validation_step(self, batch):
        input_ids, spectrogram = batch

        output = self(input_ids, spectrogram)
        loss = self.loss_function(output)

        log_dict = {
            "eval/loss": loss,
        }

        self.log_dict(log_dict, on_epoch=True)
        return log_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.optimizer_config["lr"],
        )
        return optimizer
