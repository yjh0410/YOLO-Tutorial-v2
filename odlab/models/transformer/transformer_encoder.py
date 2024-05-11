# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr

import torch
import torch.nn as nn

try:
    from .utils import get_clones, get_activation_fn
except:
    from  utils import get_clones, get_activation_fn


class TransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None):
        super().__init__()
        # -------- Basic parameters --------
        self.num_layers = num_layers
        # -------- Model parameters --------
        self.layers = get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src, src_mask, pos_embed):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask, pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 hidden_dim :int = 256,
                 num_heads  :int = 8,
                 ffn_dim    :int = 2048,
                 dropout    :float = 0.1,
                 act_type   :str   = "relu",
                 pre_norm   :bool  = False,):
        super().__init__()
        # ---------- Basic parameters ----------
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim  = ffn_dim
        self.act_type = act_type
        self.pre_norm = pre_norm
        # ---------- Model parameters ----------
        # Multi-head Self-Attn
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.dropout1  = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(hidden_dim)

        ## Feedforward network
        self.linear1    = nn.Linear(hidden_dim, ffn_dim)
        self.activation = get_activation_fn(act_type)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(ffn_dim, hidden_dim)
        self.dropout2   = nn.Dropout(dropout)
        self.norm2      = nn.LayerNorm(hidden_dim)


    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward_post(self, src, src_mask, pos_embed):
        # MSHA
        q = k = self.with_pos_embed(src, pos_embed)
        src2 = self.self_attn(q, k, src, src_key_padding_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self, src, src_mask, pos_embed):
        # MSHA
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos_embed)
        src2 = self.self_attn(q, k, src2, src_key_padding_mask=src_mask)[0]
        src = src + self.dropout1(src2)

        # FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

    def forward(self, src, src_mask, pos_embed):
        if self.pre_norm:
            return self.forward_pre(src, src_mask, pos_embed)
        else:
            return self.forward_post(src, src_mask, pos_embed)


if __name__ == "__main__":
    pass
