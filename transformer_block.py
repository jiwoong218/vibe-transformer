import torch
import torch.nn as nn

from mha import MultiHeadAttention
from ffn import FeedForwardNetwork

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        norm_x = self.norm1(x)
        attn_out = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_out)

        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)

        return x
