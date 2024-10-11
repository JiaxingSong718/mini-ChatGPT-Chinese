import os
import sys
import re
import json
import pickle
from tqdm import tqdm
from collections import OrderedDict


# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目的根目录
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from config.config import *

class BPETokenizer():
    def __init__(self) -> None:
        self.next_idx = 0
        self.byte2idx = OrderedDict()
        self.idx2byte = OrderedDict()

        # special token
        self.special_str2idx = {}
        self.special_idx2str = {}
    
    # 相邻token统计
    def _pair_stats(self,tokens,stats):
        for i in range(len(tokens)-1):
            new_token = tokens[i]+tokens[i+1]
            if new_token not in stats:
                stats[new_token] = 0
            stats[new_token] += 1

    # 合并相邻token
    def _merge_token(self,tokens,new_token):
        merged_tokens = []

        i = 0
        while i < len(tokens):
            if i+1 < len(tokens) and tokens[i] + tokens[i+1] == new_token:
                merged_tokens.append(new_token)
                i+=2

            else:
                merged_tokens.append(tokens[i])
                i+=1
        return merged_tokens
    
    def train(self,text_list,vocab_size):
        # 初始化单字节token
        for i in range(256):
            self.byte2idx[bytes([i])] = i
        self.next_idx = 256

        # 语料转byte
        tokens_list = []
        for text in text_list:
            tokens = [bytes([b]) for b in text.encode('utf-8')]
            tokens_list.append(tokens)

        # 进度条
        progress = tqdm(total=vocab_size-256)

        while True:
            # 如果词表足够大满足要求，则推出训练
            if self.next_idx >= vocab_size:
                break

            # 统计相邻token频率
            stats = {}
            for tokens in tokens_list:
                self._pair_stats(tokens,stats)

            # 如果没有更多相邻的token，无法生成更多的token，退出训练、
            if not stats:
                break

            # 合并最高频的相邻的token，作为新的token加入词表
            new_token = max(stats, key=stats.get)

            new_tokens_list = []
            for tokens in tokens_list:
                new_tokens_list.append(self._merge_token(tokens,new_token))
            tokens_list = new_tokens_list
            
            # new tokens加入词表
            self.byte2idx[new_token] = self.next_idx
            self.next_idx += 1

            # 更新进度条
            progress.update(1)

        self.idx2byte = {v:k for k,v in self.byte2idx.items()}
    
    # 词表大小
    def vocab_size(self):
        return self.next_idx
    
    # 词表
    def vocab(self):
        vocab = {}
        vocab.update(self.byte2idx)
        vocab.update({id:token.encode('utf-8') for id,token in self.special_idx2str.items()})
        return vocab
    
    # 特殊token
    def add_special_tokens(self,special_tokens):
        for token in special_tokens:
            if token not in self.special_str2idx:
                self.special_str2idx[token] = self.next_idx
                self.special_idx2str[self.next_idx] = token
                self.next_idx += 1
    def encode(self,text):
        # 特殊token分离
        pattern = '('+'|'.join([re.escape(tok) for tok in self.special_str2idx])+')'
        splits = re.split(pattern,text)

        # 编码结果
        encode_tokens = []
        encode_idx = []
        for sub_text in splits:
            if sub_text in self.special_str2idx:
                encode_idx.append(self.special_str2idx[sub_text])
                encode_tokens.append(sub_text.encode('utf-8'))

            else:
                tokens = [bytes([b]) for b in sub_text.encode('utf-8')]
                while True:
                    # 统计相邻token的频率
                    stats = {}
                    self._pair_stats(tokens,stats)

                    # 选择合并后pair最小的合并（优先合并短的）
                    new_token = None
                    for merge_token in stats:
                        if merge_token in self.byte2idx and (new_token is None or self.byte2idx[merge_token] < self.byte2idx[new_token]):
                            new_token = merge_token
                        
                    # 没有可以合并的pair
                    if new_token is None:
                        break

                    # 合并pair
                    tokens = self._merge_token(tokens,new_token)

                encode_idx.extend([self.byte2idx[token] for token in tokens])
                encode_tokens.extend(tokens)
                encode_text = [b.decode('utf-8', errors='replace') for b in encode_tokens]
        return encode_idx, encode_tokens, encode_text

    def decode(self,idx):
        bytes_list = []
        for id in idx:
            if id in self.special_idx2str:
                bytes_list.append(self.special_idx2str[id].encode('utf-8'))
            else:
                bytes_list.append(self.idx2byte[id])
        return b''.join(bytes_list).decode('utf-8',errors='replace')
    
    def save(self,file):
        with open(file, 'wb') as fp:
            fp.write(pickle.dumps((self.byte2idx,self.special_str2idx,self.next_idx)))
    def load(self,file):
        with open(file, 'rb') as fp:
            self.byte2idx,self.special_str2idx,self.next_idx = pickle.loads(fp.read())
        self.idx2byte = {v:k for k,v in self.byte2idx.items()}
        self.special_idx2str = {v:k for k,v in self.special_str2idx.items()}
