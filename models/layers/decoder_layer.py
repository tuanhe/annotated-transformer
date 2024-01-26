import torch.nn as nn
from multi_head_attention import MultiHeadedAttention
from layer_norm import LayerNorm
from position_wise_feed_forward import PositionwiseFeedForward
from utils import clones, SublayerConnection

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class DecoderLayerHubin(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadedAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(features=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.src_attn = MultiHeadedAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(features=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=ffn_hidden, dropout=drop_prob)
        self.norm3 = LayerNorm(features=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec_out, enc_out, src_mask, tgt_mask):
        # 1. compute self attention
        _x = dec_out
        x = self.self_attention(query = dec_out, key = dec_out, value = dec_out, mask=tgt_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc_out is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.src_attn(query=x, key=enc_out, value=enc_out, mask = src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
