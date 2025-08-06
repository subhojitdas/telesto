import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor] = None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), float("-1e9"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (batch, n_heads, seq_len, d_k)
        return out, attn

# Encoder part of the attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        """
        d_model: model dimension (must be divisible by num_heads)
        num_heads: number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # In the original transformer they use separate matrices; we can use combined mats for efficiency
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # final output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x):
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k)

    def _combine_heads(self, x):
        batch, num_heads, seq_len, d_k = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, num_heads, d_k)
        return x.view(batch, seq_len, num_heads * d_k)  # (batch, seq_len, d_model)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch, seq_len, _ = x.size()

        q = self.w_q(x)  # (batch, seq_len, d_model)
        k = self.w_k(x)
        v = self.w_v(x)

        # Split heads
        q = self._split_heads(q)  # (batch, num_heads, seq_len, d_k)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # prepare mask
        if mask is not None:
            # If mask is (batch, seq_len) -> make it (batch, 1, 1, seq_len)
            if mask.dim() == 2:
                # mask: 1 for positions to mask (padding), 0 for not mask
                mask = mask.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, seq_len)
            # Expand to heads if needed (broadcast will handle)

        # Attention
        attn_out, attn_weights = self.attention(q, k, v, attn_mask=mask)

        # Combine heads
        concat = self._combine_heads(attn_out)  # (batch, seq_len, d_model)

        # Final linear
        out = self.w_o(concat)  # (batch, seq_len, d_model)
        return out, attn_weights


def test_mha():
    torch.manual_seed(0)
    batch = 2
    seq_len = 5
    d_model = 32
    n_heads = 4
    x = torch.randn(batch, seq_len, d_model)

    # padding mask example: mask out position 3 and 4 for batch 1
    padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    padding_mask[1, 3:] = True  # True => masked

    mha = MultiHeadAttention(d_model=d_model, num_heads=n_heads, dropout=0.1)
    out, attn = mha(x, mask=padding_mask)

    print("out shape:", out.shape)       # (2, 5, 32)
    print("attn shape:", attn.shape)     # (2, 4, 5, 5)
    print("attn sum (per query) approx 1:", attn.sum(dim=-1)[0,0])  # sums to 1 across keys


if __name__ == '__main__':
    test_mha()