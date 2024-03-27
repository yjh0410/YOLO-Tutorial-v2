import math
import copy
import warnings
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.mlp import FFN, MLP
from ..basic.conv import LayerNorm2D, BasicConv


# ----------------- Basic Ops -----------------
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Copy from timm"""
    with torch.no_grad():
        """Copy from timm"""
        def norm_cdf(x):
            return (1. + math.erf(x / math.sqrt(2.))) / 2.

        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                        "The distribution of values may be incorrect.",
                        stacklevel=2)

        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)

        return tensor
    
def get_clones(module, N):
    if N <= 0:
        return None
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0., max=1.)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

def build_transformer(cfg, num_classes=80, return_intermediate=False):
    if cfg['transformer'] == 'plain_detr_transformer':
        return PlainDETRTransformer(d_model             = cfg['hidden_dim'],
                                    num_heads           = cfg['de_num_heads'],
                                    ffn_dim             = cfg['de_ffn_dim'],
                                    dropout             = cfg['de_dropout'],
                                    act_type            = cfg['de_act'],
                                    pre_norm            = cfg['de_pre_norm'],
                                    rpe_hidden_dim      = cfg['rpe_hidden_dim'],
                                    feature_stride      = cfg['out_stride'],
                                    num_layers          = cfg['de_num_layers'],
                                    return_intermediate = return_intermediate,
                                    use_checkpoint      = cfg['use_checkpoint'],
                                    num_queries_one2one = cfg['num_queries_one2one'],
                                    num_queries_one2many    = cfg['num_queries_one2many'],
                                    proposal_feature_levels = cfg['proposal_feature_levels'],
                                    proposal_in_stride      = cfg['out_stride'],
                                    proposal_tgt_strides    = cfg['proposal_tgt_strides'],
                                    )
    elif cfg['transformer'] == 'rtdetr_transformer':
        return RTDETRTransformer(in_dims             = cfg['backbone_feat_dims'],
                                 hidden_dim          = cfg['hidden_dim'],
                                 strides             = cfg['out_stride'],
                                 num_classes         = num_classes,
                                 num_queries         = cfg['num_queries'],
                                 num_heads           = cfg['de_num_heads'],
                                 num_layers          = cfg['de_num_layers'],
                                 num_levels          = 3,
                                 num_points          = cfg['de_num_points'],
                                 ffn_dim             = cfg['de_ffn_dim'],
                                 dropout             = cfg['de_dropout'],
                                 act_type            = cfg['de_act'],
                                 pre_norm            = cfg['de_pre_norm'],
                                 return_intermediate = return_intermediate,
                                 num_denoising       = cfg['dn_num_denoising'],
                                 label_noise_ratio   = cfg['dn_label_noise_ratio'],
                                 box_noise_scale     = cfg['dn_box_noise_scale'],
                                 learnt_init_query   = cfg['learnt_init_query'],
                                 )


# ----------------- Transformer Encoder -----------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model   :int   = 256,
                 num_heads :int   = 8,
                 ffn_dim   :int   = 1024,
                 dropout   :float = 0.1,
                 act_type  :str   = "relu",
                 pre_norm  :bool = False,
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.act_type = act_type
        self.pre_norm = pre_norm
        # ----------- Basic parameters -----------
        # Multi-head Self-Attn
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Feedforwaed Network
        self.ffn = FFN(d_model, ffn_dim, dropout, act_type)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre_norm(self, src, pos_embed):
        """
        Input:
            src:       [torch.Tensor] -> [B, N, C]
            pos_embed: [torch.Tensor] -> [B, N, C]
        Output:
            src:       [torch.Tensor] -> [B, N, C]
        """
        src = self.norm(src)
        q = k = self.with_pos_embed(src, pos_embed)

        # -------------- MHSA --------------
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout(src2)

        # -------------- FFN --------------
        src = self.ffn(src)
        
        return src

    def forward_post_norm(self, src, pos_embed):
        """
        Input:
            src:       [torch.Tensor] -> [B, N, C]
            pos_embed: [torch.Tensor] -> [B, N, C]
        Output:
            src:       [torch.Tensor] -> [B, N, C]
        """
        q = k = self.with_pos_embed(src, pos_embed)

        # -------------- MHSA --------------
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm(src)

        # -------------- FFN --------------
        src = self.ffn(src)
        
        return src

    def forward(self, src, pos_embed):
        if self.pre_norm:
            return self.forward_pre_norm(src, pos_embed)
        else:
            return self.forward_post_norm(src, pos_embed)

class TransformerEncoder(nn.Module):
    def __init__(self,
                 d_model        :int   = 256,
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 ffn_dim        :int   = 1024,
                 pe_temperature :float = 10000.,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 pre_norm       :bool  = False,
                 ):
        super().__init__()
        # ----------- Basic parameters -----------
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.act_type = act_type
        self.pre_norm = pre_norm
        self.pe_temperature = pe_temperature
        self.pos_embed = None
        # ----------- Basic parameters -----------
        self.encoder_layers = get_clones(
            TransformerEncoderLayer(d_model, num_heads, ffn_dim, dropout, act_type, pre_norm), num_layers)

    def build_2d_sincos_position_embedding(self, device, w, h, embed_dim=256, temperature=10000.):
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        
        # ----------- Check cahed pos_embed -----------
        if self.pos_embed is not None and \
            self.pos_embed.shape[2:] == [h, w]:
            return self.pos_embed
        
        # ----------- Generate grid coords -----------
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid([grid_w, grid_h])  # shape: [H, W]

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None] # shape: [N, C]
        out_h = grid_h.flatten()[..., None] @ omega[None] # shape: [N, C]

        # shape: [1, N, C]
        pos_embed = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),torch.cos(out_h)], dim=1)[None, :, :]
        pos_embed = pos_embed.to(device)
        self.pos_embed = pos_embed

        return pos_embed

    def forward(self, src):
        """
        Input:
            src:  [torch.Tensor] -> [B, C, H, W]
        Output:
            src:  [torch.Tensor] -> [B, C, H, W]
        """
        # -------- Transformer encoder --------
        channels, fmp_h, fmp_w = src.shape[1:]
        # [B, C, H, W] -> [B, N, C], N=HxW
        src_flatten = src.flatten(2).permute(0, 2, 1).contiguous()
        memory = src_flatten

        # PosEmbed: [1, N, C]
        pos_embed = self.build_2d_sincos_position_embedding(
            src.device, fmp_w, fmp_h, channels, self.pe_temperature)
        
        # Transformer Encoder layer
        for encoder in self.encoder_layers:
            memory = encoder(memory, pos_embed=pos_embed)

        # Output: [B, N, C] -> [B, C, N] -> [B, C, H, W]
        src = memory.permute(0, 2, 1).contiguous()
        src = src.view([-1, channels, fmp_h, fmp_w])

        return src
