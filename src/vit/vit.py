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
        """"""
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        #####
        # The idea of CLS token is token from BERT. This [CLS] token is learnable vector which
        # tries to learn the global representation of the image. For classification we don't need
        # a prediction per patch, we need a prediction per image. So this [CLS] token
        # wants to learn the image level representation through Transformer attention mechanism
        # ([CLS] attending to all the patches and patches are attending back to the [CLS] token )
        # In case of Avg pooling, averaging treats all patches equally. But some patches matter more
        # [CLS] lets the model learn how to weight and combine information across patches, not just average blindly.
        #####
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #####
        # A transformer’s self-attention does not know the order of its input tokens
        # If you shuffle the token sequence, self-attention treats it the same way,
        # because attention only looks at relationships between tokens, not positions.
        # That is the reason we need this learnable parameter to embed the position information of the patches
        #####
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1) # (B, P + 1, emb_dim)
        x = x + self.pos_embed

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