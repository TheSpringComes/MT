import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # w = 1.0 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        #   = exp(log(10000) * (torch.arange(0.0, d_model, 2.0) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (max_len, d_model) -> (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor):
        x = x + self.pe[:, :x.size(1)]  # (batch_size, max_len, d_model) -> (batch_size, len, d_model)
        return self.dropout(x)
    

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.001, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        self.name = "LearnablePositionalEncoding"

    def forward(self, x:torch.Tensor):
        x = x + self.pe(torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1)))
        return self.dropout(x)
    

class TransformerDecoder_model(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size = 512, num_layers=6, dropout=0.1):
        super(TransformerDecoder_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = LearnablePositionalEncoding(embed_size, dropout)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=8, dim_feedforward=hidden_size, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.name = "transformer_decoder"

    def forward(self, x: torch.Tensor):
        attn_mask = torch.full((x.size(1), x.size(1)), float('-inf'), dtype=torch.float).to(x.device)
        attn_mask = torch.triu(attn_mask, diagonal=1)  # diag>0: future mask
        key_padding_mask = torch.full((x.size(0), x.size(1)), False).to(x.device)
        key_padding_mask = key_padding_mask.masked_fill(x == 0, True)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer_decoder(x, x, tgt_mask=attn_mask, tgt_key_padding_mask=key_padding_mask)
        out = self.linear(out)
        if not self.training:
            out = out[:, -1, :]
        return out
    

class TransformerEncoderDecoder_model(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size = 512, num_layers=6, dropout=0.1):
        super(TransformerEncoderDecoder_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = LearnablePositionalEncoding(embed_size, dropout)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=8, num_encoder_layers=num_layers, 
                            num_decoder_layers=num_layers, dim_feedforward=hidden_size, dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.name = "transformer"

    def forward(self, x: torch.Tensor):
        attn_mask = torch.full((x.size(1), x.size(1)), float('-inf'), dtype=torch.float)
        attn_mask = torch.triu(attn_mask, diagonal=1).to(x.device)
        key_padding_mask = torch.full((x.size(0), x.size(1)), float(0)).to(x.device)
        key_padding_mask = key_padding_mask.masked_fill(x == 0, float('-inf'))
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer(x, x, tgt_mask=attn_mask, tgt_key_padding_mask=key_padding_mask)
        out = self.linear(out)
        if not self.training:
            out = out[:, -1, :]
        return out