from typing import Dict, Union
import itertools
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

    def forward(self, x, src_mask):
        return self.model(x, src_mask)

    def loss_function(self, output, label):
        loss = F.mse_loss(output, label)
        return loss

    def training_step(self, batch, batch_idx):
        x = batch['input_ids']
        src_mask = ~batch['attention_mask'][0].bool()
        y = batch['labels']
        
        print(f"x : {x}")
        print(f"type x : {type(x)}")
        print(f"y : {y}")
        print(f"type y : {type(y)}")
        print(f"src_mask : {src_mask}")
        print(f"type src_mask : {type(src_mask)}")

        y_hat = self(x, src_mask)
        loss = self.loss_function(y_hat, y)

        log_dict = {
            "train/loss": loss,
        }

        self.log_dict(log_dict, on_epoch=True)
        return log_dict

    def validation_step(self, batch, batch_idx):
        x = batch['input_ids']
        src_mask = ~batch['attention_mask'][0].bool()
        y = batch['labels']

        y_hat = self(x, src_mask)
        loss = self.loss_function(y_hat, y)

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
