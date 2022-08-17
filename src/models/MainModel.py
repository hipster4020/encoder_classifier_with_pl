import torch
import torch.nn as nn

from models.UtilLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        num_layers,
        num_heads,
        inner_dim,
        dropout,
        max_length=100,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_head = num_heads
        self.inner_dim = inner_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.embedding = nn.Embedding(input_dim, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_length, hidden_size)

        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(hidden_size, num_heads, inner_dim, dropout, self.device)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src,
        src_mask,
    ):
        batch_size = src.shape[0]
        seq_len = src.shape[1]

        pos = (
            torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        # embedding layer
        output = self.dropout(self.embedding(src) + self.pos_embedding(pos))

        # Dropout
        output = self.dropout(output)

        for layer in self.enc_layers:
            output = layer(output, src_mask)

        return output
