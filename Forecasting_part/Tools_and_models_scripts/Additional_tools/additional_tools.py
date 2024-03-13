import numpy as np
import torch 
from sklearn.preprocessing import StandardScaler

def all_data_scaller(train_data, val_data):
    for i in range(train_data.shape[1]):
        if i == 0:
            train_series = train_data[:,i].reshape(-1, 1)
            val_series = val_data[:,i].reshape(-1, 1)
        
            scaler_star = StandardScaler()
            train_series = scaler_star.fit_transform(train_series)
            val_series = scaler_star.transform(val_series)
        
            train_data[:, i] = train_series.flatten()
            val_data[:, i] = val_series.flatten()
        
        elif i != 0:
            train_series = train_data[:,i].reshape(-1, 1)
            val_series = val_data[:,i].reshape(-1, 1)
            
            scaler = StandardScaler()
            train_series = scaler.fit_transform(train_series)
            val_series = scaler.transform(val_series)
            
            train_data[:, i] = train_series.flatten()
            val_data[:, i] = val_series.flatten()
    return train_data, val_data, scaler_star


def sample_creator(df, lookback, horizon):

    """
    Функция нарезает данные скользящим окном, 
    lookback=исторический промежуток
    horizon=горизонт прогнозирования
    !Внимание! target переменная должна стоять в первом столбце!
    """
    X = []
    Y = np.array([np.zeros(horizon) for i in range(lookback, df.shape[0]-horizon+1 )])
    
    for i in range(lookback, df.shape[0]-horizon+1):
        x = df[i-lookback : i, :]
        X.append(x)
        y = df[i:i+horizon, 0]
        Y[i-lookback] += y
        
        
    X = torch.Tensor(np.array(X))
    Y = torch.Tensor(Y)
    return X, Y

def to_sequences(seq_size, obs, dec_seq_size):
    enc_in = []
    dec_in = []
    y = []
    for i in range(len(obs) - seq_size):
        enc_part = obs[i:(i + seq_size)]
        #dec_part = obs[ [i+seq_size-2, i+seq_size-1 ] ]
        dec_part = [obs[i+seq_size-j] for j in range(1, dec_seq_size+1) ]
        y_part = obs[i + seq_size]
        
        enc_in.append(enc_part)
        dec_in.append(dec_part)
        y.append(y_part)
        
    enc_in = torch.tensor(enc_in, dtype=torch.float32).view(-1, seq_size, 1)
    dec_in = torch.tensor(dec_in, dtype=torch.float32).view(-1, dec_seq_size, 1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return enc_in, dec_in, y
    
    
def winrate(y_true, y_pred):
    '''
    Входной формат: BatchSize*horizon(1)
    Выводит усреднённое число угаданных направлений по всему горизонту прогнозирования
    и всем батчам.
    Если горизонт = 1, то просто усредняет по батчам 
    '''
    
    diff_pred_vec = np.sign(np.diff(y_pred))
    diff_true_vec = np.sign(np.diff(y_true))
    res = np.where( diff_true_vec == diff_pred_vec , 1, 0).sum() / diff_true_vec.size
    
def winrate_long(y_true, y_pred):
    '''
    Входной формат: (BatchSize, horizon)
    Выводит усреднённое число угаданных направлений по всему горизонту прогнозирования
    и всем батчам
    '''
    
    diff_pred_vec = np.sign(np.diff(y_pred))
    diff_true_vec = np.sign( np.diff(y_true) )
    res = np.where( diff_true_vec == diff_pred_vec , 1, 0).sum() / diff_true_vec.size
    return res    
    return res    
    
  
