import hydra
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers import AutoTokenizer

from dataloader import get_dataloader, load
from models.MainModel_PL import PLEncoder


@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.tokenizer)

    # model
    model = PLEncoder(cfg.MODEL, cfg.OPTIMIZER)

    # dataloader
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)
    train_dataloader = get_dataloader(train_dataset, **cfg.DATALOADER)
    eval_dataloader = get_dataloader(eval_dataset, **cfg.DATALOADER)

    # logs
    wandb_logger = WandbLogger(**cfg.PATH.wandb)
    callbacks = [ModelCheckpoint(**cfg.PATH.ckpt)]

    trainer = Trainer(
        callbacks=callbacks,
        logger=wandb_logger,
        **cfg.TRAININGARGS,
    )
    trainer.fit(model, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
