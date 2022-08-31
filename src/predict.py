import hydra
import numpy as np
import torch
import torch.nn as nn
from pshmodule.utils import filemanager as fm
from transformers import AutoTokenizer

from dataloader import get_dataloader, load
from models.MainModel_PL import PLEncoder


@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.tokenizer)

    # model
    model = PLEncoder.load_from_checkpoint(cfg.PATH.model)
    model.cuda().eval()

    # data load
    labels = cfg.DICT.labels
    data = fm.load(cfg.PATH.test_data)

    data = tokenizer(
        data.content.tolist(),
        max_length=cfg.DATASETS.seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # predict
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    with torch.no_grad():
        pred = []
        src = data["input_ids"].cuda()
        src_mask = data["attention_mask"].cuda()

        outputs = model(src, src_mask)
        outputs = outputs.cpu().numpy()

        sigmoid = sigmoid(outputs)
        print(f"sigmoid : {sigmoid}")

    #     np_where = np.where(sigmoid > 0.8)

    #     if not np_where[0]:
    #         pred = ["etc"]
    #     else:
    #         pred = [labels[int(i)] for i in np_where[0]]

    #     print(f"pred : {pred}")


if __name__ == "__main__":
    main()
