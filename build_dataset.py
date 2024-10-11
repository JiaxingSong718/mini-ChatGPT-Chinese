import pickle
import os 
import sys 
from dataset import PreTrainDataset
from torch.utils.data import Dataset

filename='dataset.bin'

def load_dataset():
    with open(filename,'rb') as fp:
        ds=pickle.load(fp)
        return ds

class LoadDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        with open(filename,'rb') as fp:
            self.ds=pickle.load(fp)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

if __name__=='__main__':
    if os.path.exists(filename):
        ds=load_dataset()
        print(f'{filename}已存在，训练集大小：{len(ds)}，样例数据如下：')
        ids,text=ds[5]
        print(ids,text)
        sys.exit(0)

    ds=PreTrainDataset()
    with open(filename,'wb') as fp:
        ds.build_train_data()
        pickle.dump(ds,fp)
    print('dataset.bin已生成')