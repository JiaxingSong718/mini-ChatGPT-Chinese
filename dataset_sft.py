from torch.utils.data import Dataset
from Tokenizer.Tokenizer import BPETokenizer
from config.config import *
import json

class SFTDataset(Dataset):
    def __init__(self):
        self.raw_data = []
        with open('./data/sft.txt', 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  
                    self.raw_data.append(json.loads(line.strip())) 

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        tokenizer = BPETokenizer()
        tokenizer.load('./Tokenizer/tokenizer.bin')
        sample = self.raw_data[idx]
        text = f'{BOS}{IM_START}system\nYou are a helpful assistant.\n{IM_END}\n{IM_START}user\n{sample["instruction"]}\n{IM_END}\n{IM_START}assistant\n{sample["output"]}\n{IM_END}{EOS}'
        ids,_,_=tokenizer.encode(text)
        if len(ids)>SEQ_MAX_LEN-2:  # 留出BOS和EOS的token
            ids = ids[:SEQ_MAX_LEN-2]
        return (ids,text)
    
