import torch.nn as nn
from src.blocks.encoder import Encoder
from src.blocks.decoder import Decoder
from src.embedding.transformer_embedding import TransformerEmbedding
from src.blocks.generator import Generator

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,  d_model, n_head, ffn_hidden, 
                 n_layers, max_len, dropout_prob):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(d_model, n_head, ffn_hidden, n_layers, dropout_prob)
        self.decoder = Decoder(d_model, n_head, ffn_hidden, n_layers, dropout_prob)

        self.src_embed = TransformerEmbedding(d_model, src_vocab_size, max_len, dropout_prob)
        self.tgt_embed = TransformerEmbedding(d_model, tgt_vocab_size, max_len, dropout_prob)
        self.generator = Generator(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        encoder_output = self.encode(src, src_mask)
        return self.decode(encoder_output, src_mask, tgt, tgt_mask)

