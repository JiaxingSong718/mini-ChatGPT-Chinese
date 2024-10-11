from torch import nn
import torch 
from model.Embedding_and_Position import EmbeddingWithPosition
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config'))
# from config.config import GPT_BLOCKS
GPT_BLOCKS = 12

class GPT(nn.Module):
    def __init__(self,d_model,nhead,feedforward,vocab_size,seq_max_len):
        super().__init__()
        
        # positional encoding...
        self.emb=EmbeddingWithPosition(vocab_size=vocab_size,embedding_size=d_model,seq_max_len=seq_max_len)
        
        # decoder-only transformer (self-attention)
        self.dec_blocks=nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=feedforward,batch_first=True) for _ in range(GPT_BLOCKS)
        ])
        # next token probability
        self.prob_linear=nn.Linear(d_model,vocab_size)
    
    def forward(self,x,padding_mask): # x:(batch,seq)
        # 注意力遮挡
        src_mask=torch.triu(torch.ones(x.size()[1],x.size()[1]),diagonal=1).type(torch.bool).to(x.device)
        # embedding
        x=self.emb(x)
        # decoder
        for block in self.dec_blocks:
            x=block(x,src_mask=src_mask,src_key_padding_mask=padding_mask)
        # logits
        logits=self.prob_linear(x)
        return logits
