# --------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type


# ----------------------- Basic modules -----------------------
class FeedFroward(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU,
                 dropout: float = 0.0,
                 ) -> None:
        super().__init__()
        self.fc1   = nn.Linear(embedding_dim, mlp_dim)
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(mlp_dim, embedding_dim)
        self.drop2 = nn.Dropout(dropout)
        self.act   = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self,
                 in_chans    : int = 3,
                 embed_dim   : int = 768,
                 kernel_size : int = 16,
                 padding     : int = 0,
                 stride      : int = 16,
                 ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ----------------------- Model modules -----------------------
class ViTBlock(nn.Module):
    def __init__(self,
                 dim       :int,
                 num_heads :int,
                 mlp_ratio :float = 4.0,
                 qkv_bias  :bool = True,
                 act_layer :Type[nn.Module] = nn.GELU,
                 dropout   :float = 0.
                 ) -> None:
        super().__init__()
        # -------------- Model parameters --------------
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim         = dim,
                               qkv_bias    = qkv_bias,
                               num_heads   = num_heads,
                               dropout     = dropout
                               )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = FeedFroward(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        # Attention (with prenorm)
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x

        # Feedforward (with prenorm)
        x = x + self.ffn(self.norm2(x))

        return x

class Attention(nn.Module):
    def __init__(self,
                 dim       :int,
                 qkv_bias  :bool  = False,
                 num_heads :int   = 8,
                 dropout   :float = 0.
                 ):
        super().__init__()
        # --------------- Basic parameters ---------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # --------------- Network parameters ---------------
        self.qkv_proj = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        bs, N, _ = x.shape
        # ----------------- Input proj -----------------
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # ----------------- Multi-head Attn -----------------
        ## [B, N, C] -> [B, N, H, C_h] -> [B, H, N, C_h]
        q = q.view(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(bs, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        ## [B, H, Nq, C_h] X [B, H, C_h, Nk] = [B, H, Nq, Nk]
        attn = q * self.scale @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v # [B, H, Nq, C_h]

        # ----------------- Output -----------------
        x = x.permute(0, 2, 1, 3).contiguous().view(bs, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# ----------------------- Classifier -----------------------
class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        in_dim      : int,
        out_dim     : int,
        num_heads   : int = 12,
        qkv_bias    : bool = False,
        num_queries : int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = in_dim // num_heads
        self.scale = head_dim**-0.5

        self.k = nn.Linear(in_dim, in_dim, bias=qkv_bias)
        self.v = nn.Linear(in_dim, in_dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.randn(1, num_queries, in_dim) * 0.02)
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_dim, affine=False, eps=1e-6)

        self.num_queries = num_queries

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)

        out = self.linear(x_cls)

        return out, x_cls
