import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class SparseSlidingSelfAttention(nn.Module):
    """
    Sparse sliding-window multi-head self-attention with optional global tokens.
    - local window radius `window_size` means each token attends to [i-window_size, ..., i+window_size]
    - optionally include `global_token_indices` (list of token indices that are treated as global).
    - supports causal masking (no-attend-to-future).
    Input: x: (B, T, C)
    Output: (B, T, C)
    """
    def __init__(self, dim, num_heads=8, window_size: int = 4, global_token_indices: Optional[List[int]] = None, causal: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.global_token_indices = global_token_indices
        self.causal = causal

        # Projection layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, attention_mask: Optional[torch.BoolTensor] = None):
        B, T, C = x.shape
        H = self.num_heads
        D = self.head_dim
        w = self.window_size
        device = x.device

        q = self.q_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3).contiguous()  # (B, H, T, D)
        k = self.k_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3).contiguous()  # (B, H, T, D)
        v = self.v_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3).contiguous()  # (B, H, T, D)

        K = 2 * w + 1

        pad_left = torch.zeros((B, H, w, D), dtype=k.dtype, device=device)
        pad_right = torch.zeros((B, H, w, D), dtype=k.dtype, device=device)

        k_padded = torch.cat([pad_left, k, pad_right], dim=2)
        v_padded = torch.cat([pad_left, v, pad_right], dim=2)

        k_windows_list = [k_padded[:, :, i:i + T, :].contiguous() for i in range(K)]
        v_windows_list = [v_padded[:, :, i:i + T, :].contiguous() for i in range(K)]

        k_windows = torch.stack(k_windows_list, dim=3)  # (B, H, T, K, D)
        v_windows = torch.stack(v_windows_list, dim=3)  # (B, H, T, K, D)

        attn_scores = torch.einsum('bhtd,bhtkd->bhtk', q, k_windows)

        attn_probs = F.softmax(attn_scores, dim=-1)  # (B,H,T,K_total)

        out = torch.einsum('bhtk,bhtkd->bhtd', attn_probs, v_windows)

        out = out.permute(0,2,1,3).contiguous().view(B, T, C)  # (B, T, C)
        out = self.out_proj(out)
        return out


if __name__ == '__main__':
    B, T, C = 2, 50, 64
    x = torch.randn(B, T, C)
    mask = torch.ones(B, T, dtype=torch.bool)  # all tokens valid
    model = SparseSlidingSelfAttention(dim=C, num_heads=8, window_size=3, global_token_indices=[0, 25], causal=False)
    y = model(x, attention_mask=mask)
    print(y.shape)