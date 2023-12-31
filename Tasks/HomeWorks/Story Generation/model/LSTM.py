import torch
import torch.nn as nn

class LSTM_model(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size = 128, num_layers=1, dropout=0):
        super(LSTM_model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = None
        self.name = 'lstm'

    def forward(self, x:torch.Tensor, hidden = None):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        out = self.linear(out)
        if not self.training:
            out = out[:, -1, :]
        self.hidden = hidden
        return out