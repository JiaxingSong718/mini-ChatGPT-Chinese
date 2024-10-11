import os
import json
from tqdm import tqdm
from datasets import load_dataset

# 加载Zhihu-KOL数据集
ds_Zhihu = load_dataset("wangrui6/Zhihu-KOL")
# 将每个 sample 写入到 Zhihu.txt 文件中
with open('./data/Zhihu.txt', 'w', encoding='utf-8') as f:
    for sample in tqdm(ds_Zhihu['train']):
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')  # 将 sample 转换为字符串并写入文件

# 加载train_0.5M_CN数据集
ds_sft = load_dataset("BelleGroup/train_0.5M_CN")
# 将每个 sample 写入到 sft.txt 文件中
with open('./data/sft.txt', 'w', encoding='utf-8') as f:
    for sample in tqdm(ds_sft['train']):
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')  # 将 sample 转换为字符串并写入文件
