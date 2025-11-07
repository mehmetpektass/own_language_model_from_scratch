import torch
import torch.nn as nn
from torch.nn import functional as F

#parameters
batch_size = 64
block_size = 256
max_iters = 1000
eval_interval = 200
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 100
n_embd = 384
n_head = 3
n_layer = 3
droput = 0.2


torch.manual_seed(1362)

#the dataset we will use on locale
with open("input.txt", "r", encoding = "utf-8") as f:
    text = f.read()
    
    
#all the unique characters thet occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda e: [stoi[c] for c in e]
decode = lambda d: ''.join(itos[c] for c in d)


#split train and test datasets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]


#get data
def get_batch(split):
    data = train_data if split =='train' else 'test'
    ix = torch.randint(len(data) - block_size, (batch_size))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y


class LanguageModel(nn.Module):
    
    def __init__(self):
        super().__init()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)