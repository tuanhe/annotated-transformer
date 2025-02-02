import torch.nn as nn
from src.layers.layer_norm import LayerNorm

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, ln_feature_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(ln_feature_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        print("当前类的名称是：", type(self).__name__)
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))