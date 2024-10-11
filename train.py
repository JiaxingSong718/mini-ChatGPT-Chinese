import torch
import csv
from build_dataset import LoadDataset
from model.GPT import GPT
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config.config import *
from Tokenizer.Tokenizer import BPETokenizer
from tqdm import tqdm
import os


DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

# dataset=load_dataset() 
dataset = LoadDataset('dataset.bin')

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
    model=GPT(d_model=GPT_DIM,nhead=GPT_HEAD,feedforward=GPT_FF,vocab_size=tokenizer.vocab_size(),seq_max_len=SEQ_MAX_LEN).to(DEVICE)
    optimizer=torch.optim.SGD(model.parameters(),lr=3e-4,momentum=0.99)
    try:
        checkpoint=torch.load('./checkpoints/checkpoint.bin')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    except:
        checkpoint={'iter':0}
        
    dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,persistent_workers=True,collate_fn=batch_proc)
    best_loss = float('inf')
    pbar=tqdm(total=TRAIN_ITER,initial=checkpoint['iter'],postfix={'loss'})
    for i in range(checkpoint['iter'],TRAIN_ITER):
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
        
        pbar.update(1)
        pbar.set_postfix({'loss':loss.item()})

        # Save loss to CSV file
        with open('pretrain_loss.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, loss.item()])

        if i%100==0:
            checkpoint={'iter':i,'model':model.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(checkpoint,'./model_save/checkpoint.bin.tmp')
            os.replace('./model_save/checkpoint.bin.tmp','./model_save/checkpoint.bin')
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model, './model_save/Pretrain_GPT.pth')
