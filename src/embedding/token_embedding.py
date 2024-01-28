import torch
import math
import torch.nn as nn

class TokenEmbeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(TokenEmbeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    