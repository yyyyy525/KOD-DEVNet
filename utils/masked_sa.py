import torch
import torchvision
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import imageio
from torch import einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath

#  返回 t 的元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Mlp(nn.Module):
    """ Feed-forward network (FFN, a.k.a. MLP) class. """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, channels=128, mask_num=25, heads=16, patch_size=4, dropout=0.):
        super().__init__()
        project_out = not (heads == 1 and dim_head == dim)
        self.mask_num = mask_num  

        self.heads = heads  
        self.patch_size = patch_size  
        self.scale = 64 ** -0.5 

        self.attend = nn.Softmax(dim=-1)

        self.avg = nn.AdaptiveAvgPool2d((1, channels))
        self.linear = nn.Linear(channels, 1)

        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x, stage):
        global vis
        b, n, c, h, p = *x.shape, self.heads, self.patch_size  
        patch_num = int(n ** 0.5)  

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size, h=patch_num,
                      w=patch_num)
 
        q_ = self.conv_q(x)
        q = rearrange(q_, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size, h=patch_num,
                      w=patch_num)
        k_ = self.conv_k(x)
        k = rearrange(k_, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size, h=patch_num,
                      w=patch_num)
        v_ = self.conv_v(x)
        v = rearrange(v_, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size, h=patch_num,
                      w=patch_num)

        mask = q.view(b, n, p * p, c // (p * p))  
        mask = self.avg(mask).view(b * n, c // (p * p))  
        mask = self.linear(mask).view(b, n)  
        mask = self.attend(mask).clone()

        # 前 n 小
        down_k = self.mask_num
        threshold_down = torch.topk(mask, down_k, largest=False)[0][..., -1, None]  

        co_indices_to_retain = mask > threshold_down

        mask[co_indices_to_retain] = 1

        mask = mask.view(b, n, 1)  

        k_mask = k * mask
        v_mask = v * mask

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k_mask = rearrange(k_mask, 'b n (h d) -> b h n d', h=h)
        v_mask = rearrange(v_mask, 'b n (h d) -> b h n d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k_mask) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, channels=64, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, stage):
        for attn, ff in self.layers:
            x = attn(x, stage=stage) + x
            x = ff(x) + x
        return x


class MaskViT(nn.Module):
    def __init__(self, *, depth, heads, image_size=64, patch_size=4, mlp_dim=512, channels=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_size = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(patch_dim, depth, heads, mlp_dim, channels, dropout)

    def forward(self, img, stage):
        B, C, H, W = img.shape

        x = self.to_patch_embedding(img)

        x = self.transformer(x, stage)

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.patch_size, p2=self.patch_size,
                      h=H // self.patch_size, w=W // self.patch_size)
        return x
