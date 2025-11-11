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
        """
        Initializes all the necessary layers and modules for the language model.
        """
        super().__init()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Applies a custom weight initialization to Linear and Embedding layers.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        """
        Computes possibilities from token indices and calculates loss if targets are provided.
        """
        B, T = idx.shape
        
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device= device)) #(T,C)
        x = tok_emb + pos_emb
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
        
    
    def generate(self, idx, max_new_tokens):
        """
        Return the whole idx(tensor format) after creating the next token
        """
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.concat((idx, idx_next), dim=1)
        
        return idx
    
model = LanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()), 'parameters')


@torch.no_grad
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    
    return out