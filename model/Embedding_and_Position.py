from torch import nn 
import torch 
import math 

class EmbeddingWithPosition(nn.Module):
    def __init__(self, vocab_size, embedding_size, dropout=0.1, seq_max_len=2048) -> None:
        super().__init__()

        #将序列中的每一个词转为embedding向量
        self.seq_embedding = nn.Embedding(vocab_size, embedding_size)

        #为序列中每个位置准备一个位置向量，长度为embedding_size
        position_idx = torch.arange(0, seq_max_len ,dtype=torch.float).unsqueeze(-1) #(5000) -> (5000, 1)  [0, 1, 2, 3,....,4999] -> [[0], [1], [2], [3],...,[4999]]
        position_embedding_fill = position_idx * torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000.0)/embedding_size) # (seq_max_len * embedding_size/2)matrix
        position_encoding = torch.zeros(seq_max_len, embedding_size)
        position_encoding[:,0::2] = torch.sin(position_embedding_fill)
        position_encoding[:,1::2] = torch.cos(position_embedding_fill) #(seq_max_len * embedding_size)
        self.register_buffer('position_encoding', position_encoding) #位置编码不训练

        #防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):          #x:(batch_size,seq_len)
        x = self.seq_embedding(x) #(batch_size, seq_max_len, embedding_size)
        #size(position_encoding).unsqueeze(0)=(seq_max_len, embedding_size).unsqueeze(0) -> (1, seq_max_len, embedding_size) 做加法时 position_encoding 会自动升维到(batch_size, seq_max_len, embedding_size)
        x += self.position_encoding.unsqueeze(0)[:,:x.size()[1],:] #(batch_size, seq_max_len, embedding_size)
        x = self.dropout(x)
        return x

