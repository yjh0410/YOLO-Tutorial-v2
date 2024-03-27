import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- BoxRPM Cross Attention Ops -----------------
class GlobalCrossAttention(nn.Module):
    def __init__(
        self,
        dim            :int   = 256,
        num_heads      :int   = 8,
        qkv_bias       :bool  = True,
        qk_scale       :float = None,
        attn_drop      :float = 0.0,
        proj_drop      :float = 0.0,
        rpe_hidden_dim :int   = 512,
        feature_stride :int   = 16,
    ):
        super().__init__()
        # --------- Basic parameters ---------
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.feature_stride = feature_stride

        # --------- Network parameters ---------
        self.cpb_mlp1 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
        self.cpb_mlp2 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, out_dim, bias=False))
        return cpb_mlp

    def forward(self,
                query,
                reference_points,
                k_input_flatten,
                v_input_flatten,
                input_spatial_shapes,
                input_padding_mask=None,
                ):
        assert input_spatial_shapes.size(0) == 1, 'This is designed for single-scale decoder.'
        h, w = input_spatial_shapes[0]
        stride = self.feature_stride

        ref_pts = torch.cat([
            reference_points[:, :, :, :2] - reference_points[:, :, :, 2:] / 2,
            reference_points[:, :, :, :2] + reference_points[:, :, :, 2:] / 2,
        ], dim=-1)  # B, nQ, 1, 4

        pos_x = torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=w.device)[None, None, :, None] * stride  # 1, 1, w, 1
        pos_y = torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=h.device)[None, None, :, None] * stride  # 1, 1, h, 1

        delta_x = ref_pts[..., 0::2] - pos_x  # B, nQ, w, 2
        delta_y = ref_pts[..., 1::2] - pos_y  # B, nQ, h, 2

        rpe_x, rpe_y = self.cpb_mlp1(delta_x), self.cpb_mlp2(delta_y)  # B, nQ, w/h, nheads
        rpe = (rpe_x[:, :, None] + rpe_y[:, :, :, None]).flatten(2, 3) # B, nQ, h, w, nheads ->  B, nQ, h*w, nheads
        rpe = rpe.permute(0, 3, 1, 2)

        B_, N, C = k_input_flatten.shape
        k = self.k(k_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        attn += rpe
        if input_padding_mask is not None:
            attn += input_padding_mask[:, None, None] * -100

        fmin, fmax = torch.finfo(attn.dtype).min, torch.finfo(attn.dtype).max
        torch.clip_(attn, min=fmin, max=fmax)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
