import torch.nn as nn
from models.blocks.encoder import EncoderHubin
from models.blocks.decoder import DecoderHubin
from models.embedding.transformer_embedding import TransformerEmbedding
from models.blocks.generator import Generator

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        encoder_output = self.encode(src, src_mask)
        return self.decode(encoder_output, src_mask, tgt, tgt_mask)
    
class EncoderDecoderHubin(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size,  d_model, ffn_hidden, n_head, 
                 n_layers, max_len, dropout_prob):
        super(EncoderDecoderHubin, self).__init__()
        self.encoder = EncoderHubin(d_model, ffn_hidden, n_head, n_layers, dropout_prob)
        self.decoder = DecoderHubin(d_model, ffn_hidden, n_head, n_layers, dropout_prob)

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
