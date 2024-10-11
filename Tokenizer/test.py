from Tokenizer import BPETokenizer

# 加载
tokenizer = BPETokenizer()
tokenizer.load('./tokenizer.bin')
idx,tokens,text = tokenizer.encode('<|im_start|>user\n今天的天气system\n<|im_end|><|im_start|>assistant但是仍然存在一些问题<|beginoftext|>\nLife is too short to spend time with people who suck the happiness out of you. ')
print(len(idx))
print('encode:',idx,tokens)
print('================================')
print(text)

s = tokenizer.decode(idx)
print('decode:',s)