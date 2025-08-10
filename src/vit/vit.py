import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.vit.encoder_block import TransformerEncoder


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        # This convolution layer is mathematically equivalent to the
        # 1. split into patches
        # 2. flatten it into one dimensional vector with embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PathEmbedManual(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_chans * patch_size * patch_size, embed_dim, bias=True)

    def forward(self, x):
        # (B, C, H, W) => (B, num_patches, embed_dim)
        # In essence it creates an embedding for each patch.
        # If you pluck out[0, 0, :] it will give you the embedding for the batch's 1st element, 1st patch
        # If you pluck out[6, 18, :] it will give you the embedding for the batch's 7th element, 18th patch
        
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5)
        x = x.view(B, self.num_patches, C * p * p)
        x = self.proj(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # output
        out = x
        return out


if __name__ == "__main__":
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        in_chans=3,
    )

    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)