import torch.nn as nn

from models.UtilLayers import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer


# Encoder
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_size)
        self.ff_layer_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttentionLayer(
            hidden_size, n_heads, dropout, device
        )
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_size, pf_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src
