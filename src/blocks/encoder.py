import torch.nn as nn
#from copy import clones
from src.layers.layer_norm import LayerNorm
from src.layers.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, d_model, n_head, ffn_hidden, n_layers, dropout):
        super(Encoder, self).__init__()


        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  n_head=n_head,
                                                  d_ff=ffn_hidden,
                                                  dropout=dropout)
                                            for _ in range(n_layers)])
        
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
