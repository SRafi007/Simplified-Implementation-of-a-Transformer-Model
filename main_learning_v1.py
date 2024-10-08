
---

### `transformer_model.py`

```python
import torch
import torch.nn as nn
import math

# Positional Encoding Layer to give the Transformer model a sense of order in the input
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # Adding positional encoding to the input embeddings
        return x + self.encoding[:, :x.size(1), :].to(x.device)


# Multi-Head Attention Mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert (self.head_dim * num_heads == d_model), "d_model must be divisible by num_heads"
        
        # Linear layers for query, key, and value
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # Linear projections for query, key, value
        Q = self.query(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # Concatenate heads and pass through output layer
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(context)


# Transformer Block that includes multi-head attention, feed-forward layers, and normalization
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention followed by layer norm and residual connection
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward network followed by layer norm and residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# Full Transformer model that stacks multiple Transformer blocks
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, num_layers, vocab_size, max_len=512, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # Embed input tokens and add positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Pass through each transformer block
        for layer in self.layers:
            x = layer(x)

        # Output logits for each token
        return self.fc_out(x)
