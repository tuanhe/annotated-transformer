import torch.nn as nn
from src.layers.multi_head_attention import MultiHeadedAttention
from src.layers.position_wise_feed_forward import PositionwiseFeedForward
from src.layers.sublayer_connection import SublayerConnection
from src.layers.utils import clones

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
