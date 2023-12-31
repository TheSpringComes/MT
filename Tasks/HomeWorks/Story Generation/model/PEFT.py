import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter(nn.Module):
    def __init__(self, input_size=768, hidden_dim=128, dropout=0.1):
        super(Adapter, self).__init__()
        self.ff_down = nn.Linear(input_size, hidden_dim)
        self.ff_up = nn.Linear(hidden_dim, input_size)
        self.dropout = nn.Dropout(dropout)
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
        self.apply(init_weights)

    def forward(self, x):
        x = self.ff_down(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.ff_up(x)
        return x


class PrefixTuning(nn.Module):
    def __init__(self, prefix_num=20, input_size=768, encoder_hidden_size=768*2, dropout=0.1):
        super(PrefixTuning, self).__init__()
        self.prefix_num = prefix_num
        self.prefix = nn.Embedding(prefix_num, input_size)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_size, input_size),
        )
        self.prefix_prompt_tokens = torch.arange(0, prefix_num).long()

    def forward(self, x: torch.Tensor):
        prefix = self.prefix(self.prefix_prompt_tokens.to(x.device))
        prefix = self.mlp(prefix)
        x = torch.cat([prefix.expand(x.size(0), -1, -1), x], dim=1)
        return x


class LoRA(nn.Module):
    def __init__(self, output_size=768, input_size:int = 768, hidden_dim:int = 32, dropout=0.0002):
        super(LoRA, self).__init__()
        self.ff_down = nn.Linear(input_size, hidden_dim)
        self.ff_up = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
        
        def init_weights_up(m):
            if type(m) == nn.Linear:  # 对ff_down做高斯初始化
                torch.nn.init.uniform_(m.weight)
        self.ff_down.apply(init_weights_up)
        def init_weights_down(m):
            if type(m) == nn.Linear:  # 对ff_up做0初始化
                torch.nn.init.zeros_(m.weight)
        self.ff_up.apply(init_weights_down)

    def forward(self, x):
        x = self.ff_down(x)
        x = self.ff_up(self.dropout(x))
        return x