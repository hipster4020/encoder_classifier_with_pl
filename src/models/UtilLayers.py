import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_layer = nn.Linear(hid_dim, hid_dim)
        # self.fc_k = nn.Linear(hid_dim, hid_dim)
        # self.fc_v = nn.Linear(hid_dim, hid_dim)
        # self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        def transform(x):
            out = self.fc_layer(x)

            out = out.view(batch_size, -1, self.n_heads, self.head_dim)
            out = out.transpose(1, 2)  # n_batch, h, seq_len, d_k

            return out

        Q = transform(query)
        K = transform(key)
        V = transform(value)

        # calculate_attention
        batch_size, num_head, seq_len, d_k = K.size()

        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            d_k
        )  # batch_size, n_heads, seq_len, d_k

        # apply mask
        if mask is not None:
            mask = mask.view(batch_size, 1, 1, seq_len)  # batch_size, 1, 1, seq_len
            energy = energy.masked_fill(mask == 0, -1e4)

        attention_prob = F.softmax(energy, dim=-1)
        out = torch.matmul(attention_prob, V)  # batch_size, n_head, seq_len, d_k
        out = out.transpose(1, 2)  # batch_size, seq_len, n_head, d_k
        out = out.contiguous().view(
            batch_size, seq_len, self.hid_dim
        )  # batch_size, seq_len, d_model

        out = self.fc_layer(out)
        return out, attention_prob


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)

        return x
