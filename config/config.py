VOCAB_SIZE = 13000 # 词表大小
SEQ_MAX_LEN = 2048 # GPT模型输入限制

# Transformer
GPT_DIM = 768
GPT_HEAD = 12
GPT_FF = 3072
GPT_BLOCKS = 12

# training
TRAIN_ITER=80000
SFT_TRAIN_ITER=90000
BATCH_SIZE=1024

# inference
TEMPERATURE = 1.2
TOP_K = 20

# special tokens
BOS = '<|beginoftext|>'
EOS = '<|endoftext|>'
PAD = '<|padding|>'
IM_START = '<|im_start|>'
IM_END = '<|im_end|>'

# chat or generate
GPT_MODE = 'chat'