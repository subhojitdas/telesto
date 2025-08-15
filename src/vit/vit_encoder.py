import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            attention_dropout=0.1,
            projection_dropout=0.1,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # something related to xavier init (read more!)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x) # (B, N, 3*C)
        qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, N, head_dim) each
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = ( attn @ v ) # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C) # (B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(
            self,
            fan_in,
            fan_out=None,
            hidden_dim=None,
            dropout=0.1,
    ):
        super().__init__()
        fan_out = fan_out or fan_in
        hidden_dim = hidden_dim or fan_in * 4
        self.fc1 = nn.Linear(fan_in, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, fan_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
