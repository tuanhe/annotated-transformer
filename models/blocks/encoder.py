import torch.nn as nn
#from copy import clones
from models.layers.layer_norm import LayerNorm
from models.layers.encoder_layer import EncoderLayerHubin

'''
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(N)])
        
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
'''
    
class EncoderHubin(nn.Module):
    #def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
    #def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        '''
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)
        '''

        self.layers = nn.ModuleList([EncoderLayerHubin(d_model=d_model,
                                                       ffn_hidden=ffn_hidden,
                                                       n_head=n_head,
                                                       drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, enc_self_attn_mask):
        #x = self.emb(x)

        for layer in self.layers:
            x = layer(x, enc_self_attn_mask)

        return x