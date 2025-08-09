import math
import torch
import torch.nn as nn

from src.vit.attention import MultiHeadAttention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.mha(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, mask=None):
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_all.append(attn_weights)
        return x, attn_weights_all


if __name__ == '__main__':
    batch = 2
    seq_len = 5
    vocab_size = 100
    d_model = 32
    num_heads = 4
    d_ff = 64
    num_layers = 2

    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    padding_mask = torch.zeros(batch, seq_len).bool()
    padding_mask[1, 3:] = True

    encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers)
    out, attn_weights_all = encoder(input_ids, mask=padding_mask)

    print("Output shape:", out.shape)  # (2, 5, 32)
    print("Number of layers attention:", len(attn_weights_all))  # 2
    print("Attention shape of first layer:", attn_weights_all[0].shape)
