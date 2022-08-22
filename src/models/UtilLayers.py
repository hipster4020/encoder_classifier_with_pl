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

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        def transform(x, fc_layer):
            out = fc_layer(x)

            out = out.view(batch_size, -1, self.n_heads, self.head_dim)
            out = out.transpose(1, 2)  # n_batch, h, seq_len, d_k

            return out

        Q = transform(query, self.fc_q)
        K = transform(key, self.fc_k)
        V = transform(value, self.fc_v)

        # mask 수정
        print(f"mask shape : {mask.shape}")

        if mask is not None:
            mask = mask.unsqueeze(1)

        # calculate_attention
        d_k = K.size(-1)
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
        attention_prob = F.softmax(energy, dim=-1)

        print(f"attention_prob shape : {attention_prob.shape}")
        print(f"V shape : {V.shape}")

        out = torch.matmul(attention_prob, V)

        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, -1, self.head_dim)
        out = self.fc_o(out)
        return out


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
