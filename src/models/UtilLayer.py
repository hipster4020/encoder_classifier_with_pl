import torch
import torch.nn as nn
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        dropout,
        device,
    ):
        super().__init__()

        self.hidden_dim = hid_dim
        self.num_head = n_heads
        self.head_dim = self.hidden_dim // n_heads
        self.scale = torch.sqrt(torch.FloatTensor()).to(device)

        self.fcQ = nn.Linear(hid_dim, hid_dim)
        self.fcK = nn.Linear(hid_dim, hid_dim)
        self.fcV = nn.Linear(hid_dim, hid_dim)
        self.fcOut = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):
        # Scaled Dot Product Attention
        # input : (bs, seq_len, hidden_dim)
        Q = self.fcQ(query)
        K = self.fcK(key)
        V = self.fcV(value)

        Q = rearrange(
            Q,
            "bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim",
            num_head=self.num_head,
        )
        K_T = rearrange(
            K,
            "bs seq_len (num_head head_dim) -> bs num_head head_dim seq_len",
            num_head=self.num_head,
        )
        V = rearrange(
            V,
            "bs seq_len (num_head head_dim) -> bs num_head seq_len head_dim",
            num_head=self.num_head,
        )
        attention_energy = torch.matmul(Q, K_T)

        if mask is not None:
            """
            mask.shape
            if padding : (bs, 1, 1, k_len)
            if lookahead : (bs, 1, q_len, k_len)
            """
            attention_energy = torch.masked_fill(attention_energy, (mask == 0), -1e4)

        attention_energy = torch.softmax(attention_energy, dim=-1)

        result = torch.matmul(self.dropout(attention_energy), V)

        # concat
        result = rearrange(
            result, "bs num_head seq_len head_dim -> bs seq_len (num_head head_dim)"
        )
        result = self.fcOut(result)

        return result


class PositionwiseFeedforward(nn.Module):
    def __init__(
        self,
        hidden_dim,
        inner_dim,
        dropout,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim

        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        output = input
        output = self.fc1(output)
        output2 = self.relu(output)
        output2 = self.dropout(output)
        output3 = self.fc2(output2)

        return output3


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        inner_dim,
        dropout,
        device,
    ):
        super().__init__()

        self.hidden_dim = hid_dim
        self.num_head = n_heads
        self.inner_dim = inner_dim

        self.multiheadattention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.ffn = PositionwiseFeedforward(hid_dim, inner_dim, dropout)
        self.layerNorm1 = nn.LayerNorm(hid_dim)
        self.layerNorm2 = nn.LayerNorm(hid_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # input : (bs, seq_len, hidden_dim)

        # encoder attention
        output = self.multiheadattention(src, src, src, src_mask)
        output = self.dropout1(output)
        output = src + output
        output = self.layerNorm1(output)

        output_ = self.ffn(output)
        output_ = self.dropout2(output_)
        output = output + output_
        output = self.layerNorm2(output)

        # output : (bs, seq_len, hidden_dim)
        return output
