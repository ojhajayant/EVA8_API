#!/usr/bin/env python
"""
vit.py: ViT Model
"""
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('./')
dropout_value = 0.05

# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth,
                 heads, mlp_dim, pool='cls', in_channels=3, channels=32,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = channels  # We are using a Conv2d logic for patches
        # and will express patches as conv2d channels instead of taking the
        # real, direct "physical-patches" of input image.
        assert pool in {'cls',
                        'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        flattened_patch_dim = (image_height // patch_height) * (
                image_height // patch_height)  # each flattened/composed
        # -patches of  32 channels/num_patches (for ex: 256 here)
        self.to_patch_embedding = nn.Sequential(
            # Below comments -1 or b (batch_size dimension) are meant.
            # we want a conversion of input tensor: (-1, 3, 32, 32) to
            # (-1, 32, dim) for ex: dim = 512 etc. Here in the tensor
            # size: (-1,32, dim) the num_patches as 32 is the num_channels &
            # dim (for ex: 512/64 etc.) we extract (preserve) through conv2d
            # steps as given below:

            # ---------------------------------------------------------------
            # 1st: 32x32x3 |(3x3x3)x32|32x32x32
            # ---------------------------------------------------------------
            # 2nd:            GELU
            # ---------------------------------------------------------------
            # 3rd: 32x32x32|(2x2x32)x32|16x16x32 (patch_size here being: 2x2)
            # ---------------------------------------------------------------
            # 4th:            GELU
            # ---------------------------------------------------------------
            # 5th: einops.rearrange(x, 'b c h w -> b c (h w)') (i.e. for ex:
            # (-1, 32, 16, 16 )-- as obtained in the previous conv2d step
            # converts to (-1, 32, 256) (i.e. (batch_size, num_patches,
            # each flattened/composed-patches of  32 channels/num_patches)
            # ---------------------------------------------------------------
            # 6th:  einops.rearrange(x, 'b c h -> b h c 1') or x.permute(
            # 0, 2, 1).unsqueeze(-1) i.e.  permute dimensions and add a new
            # dimension for next 1x1 conv2d step i.e. after this we have the
            # tensor size as (-1, 256, 32, 1). Here, effectively we want to
            # convert the 256 dimension (i.e. the one which was the flattened
            # content of each patch, out of the 32 channels/patches, as
            # extracted  from very early conv2d steps) to a desired dimension
            # (dim) for each transformer block, hence the 256 was taken to
            # the dimension position of channels as 1x1 and up/down-move it
            # to any higher/lower (or desired dimension)
            # ---------------------------------------------------------------
            # 7th: 32x1x256 | (1x1x256)x dim |32x1 x dim  (for ex: dim = 512)
            # i.e. the tensor shape now after this 1x1 conv2d step:
            # (-1, dim , 32, 1) or for ex: with dim=512: (-1, 512, 32, 1)
            # ---------------------------------------------------------------
            # 8th:            GELU
            # ---------------------------------------------------------------
            # 9th: einops.rearrange(x, 'b h c 1 -> b c h') or  x.squeeze(
            # -1).permute(0, 2, 1) or remove last "1" so as to get a
            # tensor of size (-1, 32, dim) or for ex: for dim=512 (2, 32, 512)
            # So, finally, we will have the desired:
            # (-1, number_of_original_extracted_patches, dim) sized tensor
            # from this "to_patch_embedding" block, which would be compatible
            # with the given code (https://github.com/lucidrains/vit-pytorch/
            # blob/main/vit_pytorch/vit.py )
            # ---------------------------------------------------------------

            # 1st: 32x32x3 |(3x3x3)x32|32x32x32
            nn.Conv2d(in_channels, num_patches, (3, 3), padding=1),
            # 2nd:            GELU
            nn.GELU(),
            # 3rd: 32x32x32|(2x2x32)x32|16x16x32 (patch_size here being: 2x2)
            nn.Conv2d(num_patches, num_patches, patch_size, stride=patch_size),
            # 4th:            GELU
            nn.GELU(),
            # 5th: einops.rearrange(x, 'b c h w -> b c (h w)')
            Rearrange('b c h w -> b c (h w)'),
            # 6th:  einops.rearrange(x, 'b c h -> b h c 1')
            Rearrange('b c h -> b h c 1'),
            # 7th: 32x1x256 | (1x1x256)x dim |32x1 x dim  (for ex: dim = 512)
            nn.Conv2d(flattened_patch_dim, dim, (1, 1)),
            # 8th:            GELU
            nn.GELU(),
            # 9th: einops.rearrange(x, 'b h c 1 -> b c h')
            Rearrange('b h c 1 -> b c h'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            # nn.Conv2d(dim, num_classes, kernel_size=1)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
