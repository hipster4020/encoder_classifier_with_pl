import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TransformerLayer import EncoderLayer


class EncoderModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        num_layers,
        num_heads,
        inner_dim,
        dropout,
        num_classes,
        max_length=512,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.tok_embedding = nn.Embedding(input_dim, hidden_size)
        self.pos_embedding = nn.Embedding(max_length, hidden_size)

        self.layers = nn.ModuleList(
            [
                EncoderLayer(hidden_size, num_heads, inner_dim, dropout, self.device)
                for _ in range(num_layers)
            ]
        )
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(self.device)

        self.intermediate_layers = nn.Linear(hidden_size, num_classes)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )

        for layer in self.layers:
            src = layer(src, src_mask)

        src = src[:, 0, :]  # cls slicing
        src = F.softmax(self.intermediate_layers(src), dim=1)

        return src
