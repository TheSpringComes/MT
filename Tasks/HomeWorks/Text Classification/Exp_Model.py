import torch.nn as nn
import torch
import math

from transformers import BertModel, BertTokenizer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # w = 1.0 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        #   = exp(log(10000) * (torch.arange(0.0, d_model, 2.0) / d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (1, max_len, d_model) -> (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # (ntoken, batch_size, d_emb)
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=512, d_hid=2048, 
                 nhead=6, nlayers=6, dropout=0.2, 
                 pooling_method: str = 'mean', # 'mean', 'last', 'first', 'max'
                 multi_layer: bool = False,    # 是否使用多隐藏层作为特征的输出
                 embedding_weight=None, return_feature:bool=False):
        super(Transformer_model, self).__init__()
        self.pooling_method = pooling_method
        self.multi_layer = multi_layer
        self.return_feature = return_feature
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        """
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        """
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 transformer 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        # 请自行设计分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_emb, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 15)
        )

        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x:torch.Tensor):
        x = self.embed(x)       # (batch_size, ntoken, d_emb)  
        x = x.permute(1, 0, 2)  # (batch_size, ntoken, d_emb) -> (ntoken, batch_size, d_emb)     
        x = self.pos_encoder(x)
        if self.multi_layer:
            hidden = []
            for layer in self.transformer_encoder.layers:
                x = layer(x)
                hidden.append(x.permute(1, 0, 2))
            x = torch.stack(hidden, dim=0).sum(dim=0)  # (nlayers, ntoken, batch_size, d_emb) -> (ntoken, batch_size, d_emb)
        else:
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)  # (ntoken, batch_size, d_emb) -> (batch_size, ntoken, d_emb)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x) # 可选
        if self.pooling_method == 'mean':
            x = x.mean(dim=1)  # (batch_size, ntoken, d_emb) -> (batch_size, d_emb)
        elif self.pooling_method == 'last':
            x = x[:, -1, :]    # (batch_size, ntoken, d_emb) -> (batch_size, d_emb)
        elif self.pooling_method == 'first':
            x = x[:, 0, :]    
        elif self.pooling_method == 'max':
            x = x.max(dim=1)[0]
        else:
            raise NotImplementedError
            
        y = self.classifier(x)
        #------------------------------------------------------end------------------------------------------------------#
        if self.return_feature:
            return y, x
        return y
    
    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=100, d_hid=80, nlayers=1, dropout=0.2, 
                 pooling_method: str = 'cat', # 'mean', 'last', 'first', 'max', 'cat'
                 embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        self.pooling_method = pooling_method
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.max_len = ntoken
        self.d_hidden = d_hid
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=True, batch_first=True)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bilstm 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        # 请自行设计分类器
        if self.pooling_method == 'cat':
            self.classifier = nn.Sequential(
                nn.Linear(self.max_len*self.d_hidden*2, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 15)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.d_hidden*2, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 15)
            )
        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x:torch.Tensor):
        x = self.embed(x)
        x = self.lstm(x)[0]  # output, (_,_) -> output: (batch_size, ntoken, num_directions*hidden_size)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x)
        if self.pooling_method == 'cat':
            x = x.reshape(-1, self.max_len*self.d_hidden*2)   # ntoken*nhid*2 (2 means bidirectional)
        elif self.pooling_method == 'mean':
            x = x.mean(dim=1)  # (batch_size, ntoken, d_emb) -> (batch_size, d_emb)
        elif self.pooling_method == 'last':
            x = x[:, -1, :]
        elif self.pooling_method == 'first':
            x = x[:, 0, :]
        elif self.pooling_method == 'max':
            x = x.max(dim=1)[0]
        else:
            raise NotImplementedError
        x = self.classifier(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x
    


class Bert_model(nn.Module):
    def __init__(self, freeze_bert=False, return_feature=False,
                 pooling_method: str = 'first', # 'mean', 'last', 'first', 'max
                 multi_layer: bool = False,    # 是否使用多隐藏层作为特征的输出
                 ):
        super(Bert_model, self).__init__()
        self.return_feature = return_feature
        self.pooling_method = pooling_method
        self.multi_layer = multi_layer

        self.feature = BertModel.from_pretrained("bert-base-chinese")
        self.classifier = nn.Sequential(
            nn.Linear(768, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 15)
        )
        # 冻结 bert 的参数
        if freeze_bert:
            for param in self.feature.parameters():
                param.requires_grad = False

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.feature(x, attention_mask=mask, output_hidden_states=self.multi_layer)
        if self.multi_layer:
            x = torch.stack(x.hidden_states[1:], dim=0).sum(dim=0)  # 各隐藏层的输出之和
            if self.pooling_method == 'mean':
                x = x.mean(dim=1)  # (batch_size, ntoken, d_emb) -> (batch_size, d_emb)
            elif self.pooling_method == 'last':
                x = x[:, -1, :]
            elif self.pooling_method == 'first':
                x = x[:, 0, :]
            elif self.pooling_method == 'max':
                x = x.max(dim=1)[0]
            else:
                raise NotImplementedError
        else:
            # 选择最后一层输出的方法
            if self.pooling_method == 'first':
                x = x.pooler_output
            elif self.pooling_method == 'mean':
                x = x.last_hidden_state.mean(dim=1)
            elif self.pooling_method == 'last':
                x = x.last_hidden_state[:, -1, :]
            elif self.pooling_method == 'max':
                x = x.last_hidden_state.max(dim=1)[0]
            else:
                raise NotImplementedError
        y = self.classifier(x)
        if self.return_feature:
            return y, x
        return y