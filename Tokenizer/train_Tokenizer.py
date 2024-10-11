import os
import json
from Tokenizer import BPETokenizer
import sys
import os

text_list = []
zh = open('../data/train_zh.txt','r',encoding='utf-8').read()       
en = open('../data/train_en.txt','r',encoding='utf-8').read()
text_list = [zh,en]
# print(text_list)


# 训练
tokenizer=BPETokenizer()
tokenizer.train(text_list=text_list,vocab_size=13000)

# 特殊token
special_tokens = ['<|im_start|>','<|im_end|>','<|beginoftext|>','<|endoftext|>','<|padding|>']
tokenizer.add_special_tokens(special_tokens=special_tokens)

# 保存
tokenizer.save('tokenizer.bin')

