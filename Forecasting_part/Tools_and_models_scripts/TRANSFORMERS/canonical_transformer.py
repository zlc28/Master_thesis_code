import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    '''
    Multihead attention sublayer
    embed_size - размери эмбеддинга
    heads - кол-во головок
    head_dim - размерность эмбеддинга, которвая генерируется внутри головки 
    
    Размерность input'a (batch_size, len_sentence, embedding_size)
    '''
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size 
        self.heads = heads 
        self.head_dim = embed_size // heads 
        
        assert (self.head_dim * heads == embed_size), 'Embed size needs to be div. by heads'
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False) #Матрица значений
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False) #Матрица ключей
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False) #Матрица запросов
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size) #FC слой который идёт после конкатинации
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0] #Определяем batch size
        #Макс длина предложения.(В Encoder'е размеры совпадают)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] 
        
        #Делим embedding на 'self.heads' кусков. Делаем reshape
        #Например было (15(N), 6(макс длина предл), 512(emb size))
        #А стало (15, 6, 8, 64)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)     #Применяем linear по последнему измерению head_dim
        keys = self.keys(keys)     #Применяем linear по последнему измерению head_dim
        queries = self.queries(queries)     #Применяем linear по последнему измерению head_dim
        
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape : (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape : (N, heads, query_len, key_len) 
        # Т.е. делаем операцию Q*K.T причём с помощью einsum мы делаем всё так, чтобы
        # поменять размерности местами и оплучить размерность (N, heads, query_len, key_len) 
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.head_dim ** (1/2)), dim=3)#Normalizing over 'key_len'
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim ) # Выход размера batch_size/длина таргета/embed_size
        # attention shape : (N, heads, query_len, key_len)
        # values shape : (N, value_len, heads, head_dim)
        # shape after einsum : (N, querry_len, heads, head_dim) then flatten last two dimensions
        # and getting  (N, querry_len, embed_size)
        out = self.fc_out(out) # Прогоняем через FC слой
        return out

class TransformerBlock(nn.Module):
    '''
    Энкодер блок
    Связка Multihead attention + FFN + Skip connections + Normalization. 
    Структурная единица Энкодера. Всего в нём таких 6 штук
    '''
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) #Same as batchnorm, но усредняет не по батчу а по каждому наблюдению.
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
            )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask) #Выдаст output (N, querry_len, embed_size)
        x = self.dropout(self.norm1(attention + query)) #Skip + Norm + Dropoup
        forward = self.feed_forward(x) # FFN
        out = self.dropout(self.norm2(forward + x)) # Skip + Norm + Dropout
        return out
    

class Encoder(nn.Module):
    '''
    Энкодер целиком как ряд склеенных между собой энкодер блоков.
    src_vocab_size - размерность словаря исходного языка
    forward_expansion - сила растяжения в FFN
    max_length - максимальная длина предложения
    num_layers - количество блоков внутри энкодера(в статье их 6 штук)
    
    '''
    def __init__(
                self,
                src_vocab_size,
                embed_size,
                num_layers,
                heads,
                device,
                forward_expansion,
                dropout,
                max_length
                ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)#Создание эмбеддингов слов исходного словаря
        self.position_embedding = nn.Embedding(max_length, embed_size)#Создание позиционных эмбеддингов
        
        self.layers = nn.ModuleList( 
                    [TransformerBlock(embed_size, heads, dropout = dropout, forward_expansion=forward_expansion)
                     for _ in range(num_layers)] 
                                    ) 
        self.dropout = nn.Dropout(dropout)
        
        def forward(self, x, mask):
            N, seq_length = x.shape #N-кол-во предложений(batch size), seq_length-длина предложения(сам.длинного)
            positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)#shape(N, seq_length)
            
            out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
            # out shape : (N, seq_length, embed_size)
        
            for layer in self.layers: #Прогоняем input по 6ти блокам энкодера
                # Так как мы в Encoder layer'е, то query==key==value
                out = layer(out, out, out, mask)
            return out

class DecoderBlock(nn.Module):
    '''
    Блок декодера. На вход принимает как output энкодера, так и masked target data 
    Является стурктурной единицей Декодер слоя.
    
    '''
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, trg_mask): # x-target, value and key comes from the encoder
        attention = self.attention(x, x, x, trg_mask) #Masked Multihead sublayer (with target mask)
        query = self.dropout(self.norm(attention + x)) #Data which comes from Masked Multihead into Multihead 
        out = self.transformer_block(value, key, query, src_mack)#Multihead+FFN where value,key is encoder's output 
        return out
        
        
        
class Decoder(nn.Module):
    '''
    Декодер слой целиком.
    Придставляет из себя 'num_layers'штук декодер блоков, 
    в каждый из которых приходит как output энкодера в виде value, key матрицы, так и информация 
    снизу 'decoders data flow'
    
    trg_vocab_size - размерность словаря tqarget языка.
    
    '''
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
                    [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                     for _ in range(num_layers)]
                                    )
        
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask) #x - decoder data flow
        out = self.fc_out(x) # Классификация слова в trg vocab
        return out


        
        
class Transformer(nn.Module):
    '''
    Соединённые между собой Encoder и Decoder слои.
    Цельная модель Трансформер
    
    src_pad_idx - индекс паддинга в source словаре
    trg_pad_idx - индекс паддинга в target словаре
    '''
    def __init__(
                self,
                src_vocab_size,
                trg_vocab_size,
                src_pad_idx,
                trg_pad_idx,
                embed_size = 512,
                num_layers = 6,
                forward_expansion = 4,
                heads = 8,
                dropout = 0,
                device = "cuda",
                max_length = 100):
        
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
                src_vocab_size,
                embed_size,
                num_layers,
                heads,
                devcie,
                forward_expansion,
                dropout,
                max_length
                            )
    
        self.decoder = Decoder(
                trg_vocab_size,
                embed_size,
                num_layers,
                heads,
                forward_expansion,
                dropout,
                devcie,
                max_length
                            )
        
        self.scr_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src): #если это src_pad_idx то 0, иначе 1
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len) т.е. N'шт. boolean матриц размера 1 * src_len, где src_len это длина макс. предложения
        # Т.е. по сути мы добавили фиктивные размерности.
        return src_mask.to(self.device)
        
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len) 
        # trill делает ее треугольной! В итоге получаем N требуемых треугольных матриц с 1 и 0. Далее их как то используем
        return trg_mask.to(self.device)
    
    def forward(self, src, trg): #src_mask вообще то опциональна. А вот trg_mask обязательна.
        src_mask = self.make_src_mask(src)# Создание source маски
        trg_mask = self.make_trg_mask(trg)# Создание target маски
        enc_src = self.encoder(src, src_mask)# Подаём в энкодер src(сам батч предложений) и маску
        out = self.decoder(trg, enc_src, src_mask, trg_mask)# Подаём в дектодер trg, output энкодера, обе маски.
        return out
