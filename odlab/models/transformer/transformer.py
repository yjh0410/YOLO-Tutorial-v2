# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr

import torch
import torch.nn as nn

try:
    from .transformer_encoder import TransformerEncoderLayer, TransformerEncoder
    from .transformer_decoder import TransformerDecoderLayer, TransformerDecoder
except:
    from  transformer_encoder import TransformerEncoderLayer, TransformerEncoder
    from  transformer_decoder import TransformerDecoderLayer, TransformerDecoder


class DETRTransformer(nn.Module):
    def __init__(self,
                 hidden_dim     :int = 512,
                 num_heads      :int = 8,
                 ffn_dim        :int = 2048,
                 num_enc_layers :int = 6,
                 num_dec_layers :int = 6,
                 dropout        :float = 0.1,
                 act_type       :str   = "relu",
                 pre_norm       :bool  = False,
                 return_intermediate_dec :bool = False):
        super().__init__()
        # ---------- Basic parameters ----------
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.ffn_dim  = ffn_dim
        self.act_type = act_type
        self.pre_norm = pre_norm
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.return_intermediate_dec = return_intermediate_dec
        # ---------- Model parameters ----------
        ## Encoder module
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, num_heads, ffn_dim, dropout, act_type, pre_norm)
        encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
        self.encoder = TransformerEncoder(encoder_layer, num_enc_layers, encoder_norm)

        ## Decoder module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, num_heads, ffn_dim, dropout, act_type, pre_norm)
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(decoder_layer, num_dec_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_posembed(self, embed_dim, src_mask, temperature=10000, normalize=False):
        scale = 2 * torch.pi
        num_pos_feats = embed_dim // 2
        not_mask = ~src_mask

        # [B, H, W]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        # normalize grid coords
        if normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * scale
    
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=src_mask.device)
        dim_t_ = torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats
        dim_t = temperature ** (2 * dim_t_)

        pos_x = torch.div(x_embed[..., None], dim_t)
        pos_y = torch.div(y_embed[..., None], dim_t)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        # [B, H, W, C] -> [B, C, H, W]
        pos_embed = torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)

        return pos_embed

    def forward(self, src, src_mask, query_embed):
        # Get position embedding
        bs, c, h, w = src.shape
        pos_embed = self.get_posembed(c, src_mask, normalize=True)

        # reshape: [B, C, H, W] -> [N, B, C], H=HW
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        src_mask = src_mask.flatten(1)

        # [Nq, C] -> [Nq, B, C]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # Encoder
        memory = self.encoder(src, src_mask, pos_embed=pos_embed)

        # Decoder
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt         = tgt,
                          tgt_mask    = None,
                          memory      = memory,
                          memory_mask = src_mask,
                          memory_pos  = pos_embed,
                          query_pos   = query_embed)
        
        # [M, Nq, B, C] -> [M, B, Nq, C]
        hs = hs.transpose(1, 2)
        # [N, B, C] -> [B, C, N] -> [B, C, H, W]
        memory = memory.permute(1, 2, 0).view(bs, c, h, w)

        return hs, memory
