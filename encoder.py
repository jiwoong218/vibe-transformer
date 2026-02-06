import torch
import torch.nn as nn

from pe import PositionalEncoding
from transformer_block import TransformerBlock

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, 
                 num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
            ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, L)

        x = self.dropout(self.pos_encoding(self.embedding(x) * torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float))))

        for layer in self.layers:
            x = layer(x, mask)

        return x # (B, L, D)
