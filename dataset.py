from torch.utils.data import Dataset
from Tokenizer.Tokenizer import BPETokenizer
from config.config import *
import json
from tqdm import tqdm

class PreTrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data=[]


    def build_train_data(self):
        tokenizer = BPETokenizer()
        tokenizer.load('./Tokenizer/tokenizer.bin')

        # 知乎数据集
        self.raw_data2 = []
        with open('./data/Zhihu.txt', 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  
                    self.raw_data2.append(json.loads(line.strip())) 

        for index,sample in enumerate(tqdm(self.raw_data2, desc="Preprocess_Zhihu", ncols=150)):
            if index % 2 != 0: 
                continue
            text = '\n'.join([sample['INSTRUCTION'],sample['RESPONSE']])
            ids,_,_=tokenizer.encode(text)
            if len(ids)>SEQ_MAX_LEN-2:  # 留出BOS和EOS的token
                continue
            self.data.append((ids,text))

        # medical数据集
        with open('./data/medical.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
        header = lines.pop(0)
        for index, line in enumerate(tqdm(lines, desc='Preprocess_medcial',ncols=150)):
            if index % 5 != 0:
                continue
            if line.strip():
                parts = line.split('\t')
                if len(parts) == 4:
                    department, title, ask, answer = parts
                    entry = {
                        "title": title,
                        "ask": ask,
                        "answer": answer.strip()
                    }
                    text = '\n'.join([entry['title'],entry['ask'],entry['answer']])
                    ids,_,_=tokenizer.encode(text)
                    if len(ids)>SEQ_MAX_LEN-2:  # 留出BOS和EOS的token
                        continue
                    self.data.append((ids,text))


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
