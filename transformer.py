import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
                 src_pad_idx, trg_pad_idx, d_model=512, 
                 num_layers=6, num_heads=8, d_ff=2048, 
                 max_len=5000, dropout=0.1, device='cpu'):
        super().__init__() 
        self.encoder = Encoder(src_vocab_size, d_model, 
                               num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(trg_vocab_size, d_model, 
                               num_layers, num_heads, d_ff, max_len, dropout)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src: (batch_size, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask # (batch_size, 1, 1, src_len)

    def make_trg_mask(self, trg):
        # trg: (batch_size, trg_len)
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        
        return out
