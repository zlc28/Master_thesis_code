import numpy as np
import torch 
import torch.nn as  nn
from sklearn.preprocessing import StandardScaler


def all_data_scaler(train_data, val_data):
    
    for i in range(train_data.shape[1]):
        
        train_series = train_data[:,i].reshape(-1, 1)
        val_series = val_data[:,i].reshape(-1, 1)
        
        scaler = StandardScaler()
        train_series = scaler.fit_transform(train_series)
        val_series = scaler.transform(val_series)
        
        train_data[:, i] = train_series.flatten()
        val_data[:, i] = val_series.flatten()
    
    return train_data, val_data 

def sample_creator(df, lookback):
    '''
    Целевая перемення должна стоять в первом столбце!
    '''
    X = []
    Y = []
    # Creating sequences
    for i in range(lookback, df.shape[0]):
        x = df[i-lookback : i, :]
        X.append(x)
        y = df[i]
        Y.append(y)
        
        
    X = torch.Tensor(np.array(X))
    Y = torch.Tensor(np.array(Y))
    
    return X, Y

def winrate(y_true, y_pred):
    '''
    Входной формат: BatchSize*кол-во рядов
    Выводит усреднённое число угаданных направлений по всем активам которые пргнозируем
    и всем батчам. 
    '''
    
    diff_pred = torch.sign( torch.diff(y_pred, dim=0) )
    diff_true = torch.sign( torch.diff(y_true, dim=0) )
    
    res = torch.where( diff_pred == diff_true, 1, 0).sum().item() / diff_true.numel()
    return res


# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #zero matriz of shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #arange vector of shape (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x+self.pe[:x.size(0), :]
        return self.dropout(x)


class PatchTST(nn.Module):
    '''
    Входыне данные:
    1) Кол-во рядов т.е. фичей: (M)
    2) Длина ряда: (L)
    3) Длина патча: (P)
    4) Страйд: (S)
    5) Кол-во головок памяти: (num_heads)
    6) Внутреннаяя размерность FFN: (dim_feedforward)
    7) Кол-во слоёв в энкодер блоке: (num_layers)
    8) Размерность энкодера: (d_model)
    9) Dropout: (dropout)
    
    
    '''
    def __init__(self, M, L, P, S, num_heads, dim_feedforward, num_layers, d_model, dropout):
        super().__init__()
        
        self.M = M
        self.L = L
        self.P = P
        self.S = S
        self.N = round((L-P)/S)+1
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout = dropout
        
        self.Norm = nn.BatchNorm1d(self.M)
        self.Projection = nn.Linear(self.P, self.d_model)
        self.PosEnc = PositionalEncoding(self.d_model, dropout=self.dropout, max_len=10000)
        self.EncoderLayer = nn.TransformerEncoderLayer(d_model=self.d_model, 
                                                    nhead=self.num_heads,
                                                    dim_feedforward=self.dim_feedforward,
                                                    batch_first=True)
        self.EncoderBlock = nn.TransformerEncoder(encoder_layer=self.EncoderLayer, num_layers=self.num_layers)
        self.flatten = nn.Flatten(start_dim=-2)
        self.Task_Specific_Head = nn.Linear(self.N*self.d_model, 1)
        
                                                        
        
    def forward(self, x):                                                    # x: [B x L x M]
        x = x.permute(0, 2, 1)                                               # x: [B x M x L]
        x = self.Norm(x)                                                     # x: [B x M x L]
        x = x.unfold(dimension=-1, size=self.P, step=self.S)                 # x: [B x M x N x P]
        x = self.Projection(x)                                               # x: [B x M x N x d_model]
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))  # u: [B * M x N x d_model]
        u = u + self.PosEnc(u)                                               # u: [B * M x N x d_model]
        z = self.EncoderBlock(u)                                             # z: [B * M x N x d_model]
        z = torch.reshape(z, (-1,self.M,z.shape[-2],z.shape[-1]))            # z: [B x M x N x d_model]
        z = z.permute(0,1,3,2)                                               # z: [B x M x d_model x N]
        z = self.flatten(z)
        out = self.Task_Specific_Head(z)                                     # out: [B x M x 1]
        
        return out.reshape(-1,self.M,)