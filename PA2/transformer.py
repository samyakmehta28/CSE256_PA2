# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """

# Define global variables
batch_size = None
block_size = None
learning_rate = None
n_embd = None
n_head = None
n_layer = None
vocab_size = None
n_hidden = None
n_output = None
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
class SendVariables:
    def __init__(self, p_batch_size, p_block_size, p_learning_rate, p_n_embd, p_n_head, p_n_layer, p_vocab_size, p_n_hidden, p_n_output):
        # Assign values to global variables
        global batch_size
        global block_size
        global learning_rate
        global n_embd
        global n_head
        global n_layer
        global vocab_size
        global n_hidden
        global n_output
        
        batch_size = p_batch_size
        block_size = p_block_size
        learning_rate = p_learning_rate
        n_embd = p_n_embd
        n_head = p_n_head
        n_layer = p_n_layer
        vocab_size = p_vocab_size
        n_hidden = p_n_hidden
        n_output = p_n_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size,type_pos_emb, mask=True, num_heads=1,i=0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_num = i
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.mask=mask
        self.type_pos_emb = type_pos_emb
        self.attn_map = None

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        attention_map = (q @ k.transpose(-2,-1)) * k.shape[-1]**-0.5
        if self.type_pos_emb == 'AliBi':
            #('AliBi')
            base_bias = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1) 
            m = base_bias*(2**(-8*(self.head_num+1)/self.num_heads))
            attention_map = (q @ k.transpose(-2,-1)+ m) * k.shape[-1]**-0.5 
        if self.mask:
            attention_map = attention_map.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        attention_map = F.softmax(attention_map, dim=-1) 
        self.attn_map = attention_map
        attention_map = self.dropout(attention_map)
        v = self.value(x) # (B,T,hs)
        out = attention_map @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, type_pos_emb, mask=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size,type_pos_emb,mask,num_heads, i) for i in range(num_heads)])
        self.project = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = torch.cat([h(x) for h in self.heads], dim=-1)
        res = self.dropout(self.project(res))
        return res

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head,type_pos_emb,mask=True):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size,type_pos_emb,mask)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.type_pos_emb = type_pos_emb

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderModel(nn.Module):

    def __init__(self, type_pos_emb='absolute'):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        if type_pos_emb == 'absolute':
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, type_pos_emb) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.type_pos_emb = type_pos_emb

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x=tok_emb
        if self.type_pos_emb == 'absolute':
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = x + pos_emb
        
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        attn_map = self.blocks[-1].sa.heads[-1].attn_map
        return logits, loss, attn_map



class Encoder(nn.Module):

    def __init__(self, type_pos_emb='absolute'):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.type_pos_emb = type_pos_emb
        if type_pos_emb == 'absolute':
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, type_pos_emb, mask=False) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        #pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        #x = tok_emb + pos_emb # (B,T,C)
        x = tok_emb
        if self.type_pos_emb == 'absolute':
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = x + pos_emb
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)

        attn_map = self.blocks[-1].sa.heads[-1].attn_map
        return x, attn_map


class Classifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)
    
class EncoderModel(nn.Module):
    def __init__(self, type_pos_emb='absolute'):
        super(EncoderModel, self).__init__()
        self.encoder = Encoder(type_pos_emb)
        self.classifier = Classifier(n_embd, n_hidden, n_output)

    def forward(self, x):
        x,attn_map = self.encoder(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x,attn_map,attn_map
