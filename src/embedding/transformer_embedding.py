import torch
import math
import torch.nn as nn
from src.embedding.positional_encoding import PositionalEncoding
from src.embedding.token_embedding import TokenEmbeddings


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, d_model, vocab_size, max_len, dropout_prob):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbeddings(d_model, vocab_size)
        self.pos_emb = PositionalEncoding(d_model, dropout_prob, max_len)

    def forward(self, x):
        tok_emb_out = self.tok_emb(x)
        return self.pos_emb(tok_emb_out)