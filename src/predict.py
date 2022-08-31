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
    df = fm.load(cfg.PATH.test_data)

    data = tokenizer(
        df.content.astype(str).values.tolist(),
        max_length=cfg.DATASETS.seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # predict
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pred = []
    with torch.no_grad():
        src = data["input_ids"].cuda()
        src_mask = data["attention_mask"].cuda()

        outputs = model(src, src_mask)
        outputs = outputs.cpu().numpy()

        sigmoid = sigmoid(outputs)
        np_where = [np.where(i > 0.8)[0].tolist() for i in sigmoid]

        for v in np_where:
            value = []
            if not v:
                value.append("etc")
            else:
                for i in v:
                    value.append(labels[i])
            pred.append(value)

    df["predict"] = pred
    fm.save(cfg.PATH.check_data, df)


if __name__ == "__main__":
    main()
