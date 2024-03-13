import numpy as np
import torch 
import torch.nn as  nn
from sklearn.preprocessing import StandardScaler


class TimeMixer(nn.Module):
    def __init__(self, seq_len, num_features, dropout):
        super(TimeMixer, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.dropout = dropout
        
        self.Norm1 = nn.BatchNorm1d(self.num_features)
        self.MLPtime = nn.Linear(self.seq_len, self.seq_len)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)
    
    def forward(self, x):
        copy = x
        x = self.Norm1(x.transpose(1,2))
        x = self.Relu(self.MLPtime(x))
        x = self.dropout(x).transpose(1,2)
        x = x + copy
        return x

class FeatureMixer(nn.Module):
    def __init__(self, seq_len, num_features, dropout):
        super(FeatureMixer, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.dropout = dropout
        
        self.Norm1 = nn.BatchNorm1d(self.num_features)
        self.MLPfeat1 = nn.Linear(self.num_features, self.num_features)
        self.MLPfeat2 = nn.Linear(self.num_features, self.num_features)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)
    def forward(self, x):
        copy = x
        x = self.Norm1(x.transpose(1,2)).transpose(1,2)
        x = self.MLPfeat1(x)
        x = self.Relu(x)
        x = self.dropout(x)
        x = self.MLPfeat2(x)
        x = self.dropout(x)
        x = copy + x
        return x

class MixerLayer(nn.Module):
    def __init__(self, seq_len, num_features, dropout):
        super(MixerLayer, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.dropout = dropout
        
        self.TimeMixer = TimeMixer(self.seq_len, self.num_features, self.dropout)
        self.FeatureMixer = FeatureMixer(self.seq_len, self.num_features, self.dropout)
    def forward(self, x):
        x = self.TimeMixer(x)
        x = self.FeatureMixer(x)
        return x

class TemporalProjection(nn.Module):
    def __init__(self, seq_len, num_features, forecast_horizon):
        super(TemporalProjection, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.forecast_horizon = forecast_horizon
        self.FC = nn.Linear(self.seq_len, self.forecast_horizon)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.FC(x)
        x = x.transpose(1,2)
        return x
    
class TSMixer(nn.Module):
    '''
    seq_len = длина последовательности, на которую смотрит модель для прогноза
    num_features = количество признаков
    forecast_horizon = горизонт прогнозирвания
    dropout = коэффициент дропаута
    num_of_blocks = количество Mixer блоков в архитектуре
    '''
    def __init__(self, seq_len, num_features, forecast_horizon, dropout, num_of_blocks):
        super(TSMixer, self).__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.forecast_horizon = forecast_horizon
        self.num_of_blocks = num_of_blocks
        self.dropout = dropout
        
        self.MixerBlock = nn.ModuleList( 
                    [MixerLayer(self.seq_len, self.num_features, self.dropout) for _ in range(self.num_of_blocks)] 
                                    )
        self.TemporalProjection = TemporalProjection(self.seq_len, self.num_features, self.forecast_horizon)
    
    def forward(self, x):
        for layer in self.MixerBlock:
            x = layer(x)
        x = self.TemporalProjection(x)
        x = x[:,:,0]
        return x
        
        
