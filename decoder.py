import torch
import torch.nn as nn
from pe import PositionalEncoding
from mha import MultiHeadAttention
from ffn import FeedForwardNetwork

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        norm_x = self.norm1(x)
        attn_out = self.self_attn(norm_x, norm_x, norm_x, trg_mask)
        x = x + self.dropout(attn_out)

        norm_x = self.norm2(x)
        attn_out = self.cross_attn(norm_x, enc_out, enc_out, src_mask)
        x = x + self.dropout(attn_out)

        norm_x = self.norm3(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, 
                 d_ff, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
            ])

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # x: (B, trg_seq_len)
        x = self.dropout(self.pos_encoding(self.embedding(x) * torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float))))

        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)

        return self.fc_out(x) # (batch_size, trg_seq_len, vocab_size)
