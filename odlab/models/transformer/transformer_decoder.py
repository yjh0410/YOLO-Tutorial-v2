# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr

import torch
import torch.nn as nn

try:
    from .utils import get_clones, get_activation_fn
except:
    from  utils import get_clones, get_activation_fn


class TransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super().__init__()
        # --------- Basic parameters ---------
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # --------- Model parameters ---------
        self.layers = get_clones(decoder_layer, num_layers)
        self.norm = norm

    def forward(self,
                tgt,
                tgt_mask,
                memory,
                memory_mask,
                memory_pos,
                query_pos):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output,
                           tgt_mask,
                           memory,
                           memory_mask,
                           memory_pos,
                           query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)   # [M, N, B, C]

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 ffn_dim=2048,
                 dropout=0.1,
                 act_type="relu",
                 pre_norm=False):
        super().__init__()
        # ---------- Basic parameters ----------
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim  = ffn_dim
        self.act_type = act_type
        self.pre_norm = pre_norm
        # ---------- Model parameters ----------
        ## MHSA for object queries
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.dropout1  = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(hidden_dim)

        ## MHCA for object queries
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2    = nn.LayerNorm(hidden_dim)

        ## Feedforward network
        self.linear1    = nn.Linear(hidden_dim, ffn_dim)
        self.activation = get_activation_fn(act_type)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(ffn_dim, hidden_dim)
        self.dropout3   = nn.Dropout(dropout)
        self.norm3      = nn.LayerNorm(hidden_dim)

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward_post(self,
                     tgt,
                     tgt_mask,
                     memory,
                     memory_mask,
                     memory_pos,
                     query_pos,
                     ):
        # MHSA for object queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # MHCA between object queries and image features
        q = self.with_pos_embed(tgt, query_pos)
        k = self.with_pos_embed(memory, memory_pos)
        tgt2 = self.multihead_attn(q, k, memory, key_padding_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self,
                    tgt,
                    tgt_mask,
                    memory,
                    memory_mask,
                    memory_pos,
                    query_pos,
                    ):
        # MHSA for object queries
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, tgt2, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        # MHCA between object queries and image features
        q = self.with_pos_embed(tgt2, query_pos)
        k = self.with_pos_embed(memory, memory_pos)
        tgt2 = self.multihead_attn(q, k, memory, key_padding_mask=memory_mask)[0]
        tgt = tgt + self.dropout2(tgt2)

        # FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self,
                tgt,
                tgt_mask,
                memory,
                memory_mask,
                memory_pos,
                query_pos,):
        if self.pre_norm:
            return self.forward_pre(tgt, tgt_mask, memory, memory_mask, memory_pos, query_pos)
        else:
            return self.forward_post(tgt, tgt_mask, memory, memory_mask, memory_pos, query_pos)


if __name__ == "__main__":
    pass
