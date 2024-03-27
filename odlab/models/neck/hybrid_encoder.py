from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.conv import BasicConv, RepCSPLayer
from ..basic.transformer import TransformerEncoder


# -------------- Feature Pyramid Network + Transformer Encoder --------------
class HybridEncoder(nn.Module):
    def __init__(self, 
                 in_dims        :List  = [256, 512, 1024],
                 out_dim        :int   = 256,
                 num_blocks     :int   = 3,
                 expansion      :float = 1.0,
                 act_type       :str   = 'silu',
                 norm_type      :str   = 'GN',
                 depthwise      :bool  = False,
                 # Transformer's parameters
                 num_heads      :int   = 8,
                 num_layers     :int   = 1,
                 ffn_dim        :int   = 1024,
                 dropout        :float = 0.1,
                 pe_temperature :float = 10000.,
                 en_act_type    :str   = 'gelu',
                 en_pre_norm    :bool  = False,
                 ) -> None:
        super(HybridEncoder, self).__init__()
        # ---------------- Basic parameters ----------------
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.out_dims = [self.out_dim] * len(in_dims)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        c3, c4, c5 = in_dims

        # ---------------- Input projs ----------------
        self.input_proj_1 = BasicConv(c5, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.input_proj_2 = BasicConv(c4, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.input_proj_3 = BasicConv(c3, self.out_dim, kernel_size=1, act_type=act_type, norm_type=norm_type)

        # ---------------- Transformer Encoder ----------------
        self.transformer_encoder = TransformerEncoder(d_model        = self.out_dim,
                                                      num_heads      = num_heads,
                                                      num_layers     = num_layers,
                                                      ffn_dim        = ffn_dim,
                                                      pe_temperature = pe_temperature,
                                                      dropout        = dropout,
                                                      act_type       = en_act_type,
                                                      pre_norm       = en_pre_norm,
                                                      )

        # ---------------- Top dwon FPN ----------------
        ## P5 -> P4
        self.reduce_layer_1 = BasicConv(self.out_dim, self.out_dim,
                                        kernel_size=1, padding=0, stride=1,
                                        act_type=act_type, norm_type=norm_type)
        self.top_down_layer_1 = RepCSPLayer(in_dim      = self.out_dim * 2,
                                            out_dim     = self.out_dim,
                                            num_blocks  = num_blocks,
                                            expansion   = expansion,
                                            act_type    = act_type,
                                            norm_type   = norm_type,
                                            )
        ## P4 -> P3
        self.reduce_layer_2 = BasicConv(self.out_dim, self.out_dim,
                                        kernel_size=1, padding=0, stride=1,
                                        act_type=act_type, norm_type=norm_type)
        self.top_down_layer_2 = RepCSPLayer(in_dim      = self.out_dim * 2,
                                            out_dim     = self.out_dim,
                                            num_blocks  = num_blocks,
                                            expansion   = expansion,
                                            act_type    = act_type,
                                            norm_type   = norm_type,
                                            )
        
        # ---------------- Bottom up PAN----------------
        ## P3 -> P4
        self.dowmsample_layer_1 = BasicConv(self.out_dim, self.out_dim,
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_1 = RepCSPLayer(in_dim      = self.out_dim * 2,
                                             out_dim     = self.out_dim,
                                             num_blocks  = num_blocks,
                                             expansion   = expansion,
                                             act_type    = act_type,
                                             norm_type   = norm_type,
                                             )
        ## P4 -> P5
        self.dowmsample_layer_2 = BasicConv(self.out_dim, self.out_dim,
                                            kernel_size=3, padding=1, stride=2,
                                            act_type=act_type, norm_type=norm_type, depthwise=depthwise)
        self.bottom_up_layer_2 = RepCSPLayer(in_dim      = self.out_dim * 2,
                                             out_dim     = self.out_dim,
                                             num_blocks  = num_blocks,
                                             expansion   = expansion,
                                             act_type    = act_type,
                                             norm_type   = norm_type,
                                             )

        self.init_weights()
  
    def init_weights(self):
        """Initialize the parameters."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()

    def forward(self, features):
        c3, c4, c5 = features

        # -------- Input projs --------
        p5 = self.input_proj_1(c5)
        p4 = self.input_proj_2(c4)
        p3 = self.input_proj_3(c3)

        # -------- Transformer encoder --------
        p5 = self.transformer_encoder(p5)

        # -------- Top down FPN --------
        p5_in = self.reduce_layer_1(p5)
        p5_up = F.interpolate(p5_in, size=p4.shape[2:])
        p4 = self.top_down_layer_1(torch.cat([p4, p5_up], dim=1))

        p4_in = self.reduce_layer_2(p4)
        p4_up = F.interpolate(p4_in, size=p3.shape[2:])
        p3 = self.top_down_layer_2(torch.cat([p3, p4_up], dim=1))

        # -------- Bottom up PAN --------
        p3_ds = self.dowmsample_layer_1(p3)
        p4 = self.bottom_up_layer_1(torch.cat([p4_in, p3_ds], dim=1))

        p4_ds = self.dowmsample_layer_2(p4)
        p5 = self.bottom_up_layer_2(torch.cat([p5_in, p4_ds], dim=1))

        out_feats = [p3, p4, p5]
        
        return out_feats
