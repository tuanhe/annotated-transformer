import torch.nn as nn
#from copy import clones
from src.layers.layer_norm import LayerNorm
from src.layers.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, d_model, n_head, ffn_hidden, n_layers, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  n_head=n_head,
                                                  d_ff=ffn_hidden,
                                                  dropout=dropout)
                                            for _ in range(n_layers)])

        self.norm = LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)