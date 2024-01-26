import torch
import math
import torch.nn as nn
from models.embedding.positional_encoding import PositionalEncodingHubin
from models.embedding.token_embedding import EmbeddingsHubin


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
        self.tok_emb = EmbeddingsHubin(d_model, vocab_size)
        self.pos_emb = PositionalEncodingHubin(d_model, dropout_prob, max_len)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        return self.pos_emb(tok_emb)