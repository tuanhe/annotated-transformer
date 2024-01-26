import torch.nn as nn
from copy import clones
from models.layers.layer_norm import LayerNorm
from models.layers.decoder_layer import DecoderLayerHubin

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class DecoderHubin(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
    #def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(DecoderHubin, self).__init__()
        '''
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)
        '''
        self.layers = nn.ModuleList([DecoderLayerHubin(d_model=d_model,
                                                       ffn_hidden=ffn_hidden,
                                                       n_head=n_head,
                                                       drop_prob=drop_prob)
                                                       for _ in range(n_layers)])

        #self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, x, enc_outputs, src_mask, tgt_mask):
        #dec_outputs = self.emb(dec_inputs)

        for layer in self.layers:
            x = layer(x, enc_outputs, src_mask, tgt_mask)

        # pass to LM head
        #output = self.linear(dec_outputs)
        #print(" checkout the output shape:", output.shape)
        return x