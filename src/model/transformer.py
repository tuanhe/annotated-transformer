import torch.nn as nn
from src.model.encoder_decoder import EncoderDecoder
#from models.

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, ffn_hidden=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    model = EncoderDecoder(src_vocab_size = src_vocab, 
                           tgt_vocab_size = tgt_vocab,  
                           d_model = d_model, 
                           n_head = h, 
                           ffn_hidden = ffn_hidden,
                           n_layers = N, 
                           max_len = 5000, 
                           dropout_prob = dropout)

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model