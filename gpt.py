import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # embedding size
n_head = 6
n_layer = 6
dropout = 0.2
# ---------------------------

torch.manual_seed(1337)

# download the Shakespeare dataset if needed
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read text input in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# Here all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder_ take a list of integer, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset the block by 1
  x, y = x.to(device), y.to(device) # send data to gpu
  return x, y 

class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # as tril is not a trainable parameter, we use the register buffer from pytorch
        # to save the tril as constant for the model
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout) # dropout for regularization

    def forward(self, x):
        B,T,C = x.shape # batch, time, channels
        # C = n_embd
        # head_size = H
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)
        H = k.shape[-1]
        # compute attention scores ("affinities")
        # by multiplying every q for every k using mat mul
        # transpose last 2 dimensions from k so that k = (B, H, T)
        # normalize by dividing by sqrt of H
        wei = q @ k.transpose(-2, -1) * H**-0.5 # (B, T, H) @ (B, H, T) ----> (B, T, T)
        # use only lower triangular part of wei
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # softmax exponentiates and normalizes, making each row sum 1 and convert -inf to 0
        wei = F.softmax(wei, dim=-1) # softmax along last dimension as it has batch dim as well
        # now wei has the weights of each element at each step of T, rows add to 1 (not uniform)
        wei = self.dropout(wei) # regularization
        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, H)
        out = wei @ v # (B, T, T) @ (B, T, H) -> (B, T, H)
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # the heads is a list of head modules
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # add a projection linear layer for the output that will be added to the residual pathway
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) # dropout layer for regularization

    def forward(self, x):
        # return the concatenation of the results from all heads over the channel dimension
        # we use head_size H = C / num_heads so the resulting dimension is C
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, H*num_heads=C)
        out = self.dropout(self.proj(out)) # (B, T, C)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # the paper shows an internal layer size 4 times the input for this layer (section 3.3)
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer with no ReLU
            nn.Dropout(dropout), # dropout layer for regularization
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of attention heads we'd like
        super().__init__()
        # H = C // num_heads so that the final dimension of all heads is C
        head_size = n_embd // n_head
        # communication: multi headed self attention
        self.sa = MultiHeadAttention(n_head, head_size)
        # computation: feed forward layer
        self.ffwd = FeedForward(n_embd)
        # 2 layer norm layers: 0 mean and 1 std in the channel dimension per token (row)
        self.ln1 = nn.LayerNorm(n_embd) # used after multi headed attention
        self.ln2 = nn.LayerNorm(n_embd) # used after feed forward

    def forward(self, x):
        # we add x as residual connections
        # layer norm is applied BEFORE the residual connection, 
        # which is different from the original paper, which uses layernorm after adding the res conn
        x = x + self.sa(self.ln1(x)) 
        x = x +  self.ffwd(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # positional embeddings
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    # transformer blocks (n_layer number of blocks with n_head heads each)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    # language modeling head to get output
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    # idx and targets are both (B,T) tensor of integers
    tok_emb = self.token_embedding_table(idx) # (B,T,C) C = n_embd
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
    # add token and positional embeddings
    x = tok_emb + pos_emb # (B,T,C)
    x = self.blocks(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
  # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens
      # if we have more than block size as input, the positional embeddings will crash
      idx_cond = idx[:, -block_size:] # (B, T<=block_size)
      # get the predictions
      logits, loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=1) # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running squence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = TransformerLanguageModel()
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'Million parameters')

# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))
open('generated.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))