import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# 1. Training text
text = """
machine learning is interesting.
deep learning is powerful.
artificial intelligence is the future.
language models learn from text.
transformers are used in modern ai.
""" * 100
# 2. Character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi ={ c : i for i,c in enumerate(chars) }
itos ={ i : c for c,i in stoi.items() }

data = torch.tensor(
    [stoi[c] for c in text],
    dtype=torch.long
)
# 3. Model configuration
batch_size =32
block_size =32
embed_dim =64
num_heads =4
num_layers=2
learning_rate =0.001
training_steps =500

# 4. Get a training batch
def get_batch():
    starts = torch.randint(
        0,
        len(data)-block_size-1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i+block_size] for i in starts
    ])

    y =torch.stack([
        data[i+1:i+block_size+1] for i in starts
    ])

    return x,y

# 5. GPT block
class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim,4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim,embed_dim)
        )

    def forward(self,x):
        seq_len = x.size(1)
        # Block access to future positions(mask)
        mask = torch.triu(
            torch.ones(
                seq_len,seq_len,
                device=x.device,
                dtype = torch.bool
            ),
            diagonal=1
        )

        attention_output,_=self.attention(
            x,x,x,
            attn_mask = mask,
            need_weights =False
        )

        x = self.norm1(x+attention_output)
        x = self.norm2(x+self.ffn(x))

        return x

# 6. Mini GPT model

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size,embed_dim)
        self.position_embedding = nn.Embedding(block_size,embed_dim)
        # self.blocks = nn.Sequential(*[GPTBlock()for _ in range(num_layers)])
        self.blocks = nn.Sequential(
            *[GPTBlock() for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_layer = nn.Linear(embed_dim,vocab_size)

    def forward(self,idx,targts =None):
        batch,seq_len = idx.shape
        token_emb = self.token_embedding(idx)
        positions = torch.arange(seq_len,device=idx.device)
        position_emb = self.position_embedding(positions)
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.output_layer(x)

        loss = None

        if targts is not None:
            loss = F.cross_entropy(
                logits.reshape(-1,vocab_size),
                targts.reshape(-1)
            )
        return logits,loss

    @torch.no_grad()
    def generate(self,idx,max_new_tokens):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:,-block_size:]
            logits,_=self(idx_cond)
            # Get scores for the last position
            logits = logits[:,-1,:]
            probabilities = F.softmax(logits,dim=1)
            next_token = torch.multinomial(probabilities,num_samples=1)
            idx = torch.cat((idx,next_token),dim=1)

        return idx
    
# 7. Initialize model and optimizer
model = MiniGPT()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print("Vocabulary size:", vocab_size)
print("Model parameters:",
      sum(p.numel() for p in model.parameters()))

# 8. Training
model.train()

for step in range(training_steps):
    x, y = get_batch()

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(
            f"Step {step}, Loss: {loss.item():.4f}"
        )

# 9. Generate text
start = torch.zeros((1, 1), dtype=torch.long)

generated = model.generate(
    start,
    max_new_tokens=100
)

generated_text = "".join(
    itos[token.item()] for token in generated[0]
)

print("\nGenerated text:")
print(generated_text)
