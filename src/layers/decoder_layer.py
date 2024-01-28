import torch.nn as nn
from src.layers.multi_head_attention import MultiHeadedAttention
from src.layers.position_wise_feed_forward import PositionwiseFeedForward
from src.layers.sublayer_connection import SublayerConnection
from src.layers.utils import clones

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, n_head, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadedAttention(n_head, d_model, dropout)
        self.src_attn = MultiHeadedAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)