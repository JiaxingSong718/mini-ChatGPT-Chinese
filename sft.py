import torch
import csv
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from build_dataset import LoadSFTDataset
from model.GPT import GPT
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config.config import *
from Tokenizer.Tokenizer import BPETokenizer
from tqdm import tqdm
import os 

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

dataset=LoadSFTDataset()

tokenizer=BPETokenizer()
tokenizer.load('./Tokenizer/tokenizer.bin')
pad_ids,_,_=tokenizer.encode(PAD)
 
def batch_proc(batch):
    bos_ids,_,_=tokenizer.encode(BOS)
    eos_ids,_,_=tokenizer.encode(EOS)
    pad_ids,_,_=tokenizer.encode(PAD)
    
    batch_x=[]
    batch_chatml=[]
    # bpe encode
    for sample in batch:
        ids,chatml=sample
        ids=bos_ids+ids+eos_ids
        batch_x.append(ids)
        batch_chatml.append(chatml)
    
    # padding
    max_len=max([len(ids) for ids in batch_x])
    for ids in batch_x:
        if len(ids)<max_len:
            ids.extend(pad_ids*(max_len-len(ids)))
    batch_x=torch.tensor(batch_x,dtype=torch.long)
    
    # padding mask
    batch_padding_mask=(batch_x==pad_ids[0])
    return batch_x,batch_padding_mask,batch_chatml

if __name__=='__main__':
    model=GPT(d_model=GPT_DIM,nhead=GPT_HEAD,feedforward=GPT_FF,vocab_size=tokenizer.vocab_size(),seq_max_len=SEQ_MAX_LEN).to(DEVICE) # 模型
    initial_lr = 5e-5
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    # 学习率调度器：指数衰减
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    min_lr = 1e-7
        
    model = torch.load('./model_save/SFT_GPT.pth')
    print('load sucessfully!')
        
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,collate_fn=batch_proc)
    best_loss = float('inf')
    progress_bar = tqdm(range(SFT_TRAIN_ITER), ncols=200)
    for i in progress_bar:
        batch_ids,batch_padding_mask,batch_chatml=next(iter(dataloader))

        batch_ids=batch_ids.to(DEVICE)
        batch_padding_mask=batch_padding_mask.to(DEVICE)
        
        logtis=model(batch_ids,batch_padding_mask)  # (batch,seq,vocab)
        
        probs=logtis[:,:-1,:]   # (batch,seq-1,vocab)
        targets=batch_ids[:,1:] # (batch,seq-1)
        loss=F.cross_entropy(probs.reshape(-1,probs.size(2)),targets.reshape(-1),ignore_index=pad_ids[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 更新学习率
        scheduler.step()
        # 获取当前学习率并确保不低于最小值
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < min_lr:
            current_lr = min_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        progress_bar.set_postfix(loss=loss.item())
        # Save loss to CSV file
        with open('sft_loss.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, loss.item()])

        if i%100==0:
            checkpoint={'iter':i,'model':model.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(checkpoint,'./model_save/sft.bin.tmp')
            os.replace('./model_save/sft.bin.tmp','./model_save/sft.bin')
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model, './model_save/SFT_GPT.pth')