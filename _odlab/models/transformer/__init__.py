from .transformer import DETRTransformer


def build_transformer(cfg, return_intermediate_dec):
    if cfg.transformer == "detr_transformer":
        return DETRTransformer(hidden_dim     = cfg.hidden_dim,
                               num_heads      = cfg.num_heads,
                               ffn_dim        = cfg.feedforward_dim,
                               num_enc_layers = cfg.num_enc_layers,
                               num_dec_layers = cfg.num_dec_layers,
                               dropout        = cfg.dropout,
                               act_type       = cfg.tr_act,
                               pre_norm       = cfg.pre_norm,
                               return_intermediate_dec=return_intermediate_dec)
    else:
        raise NotImplementedError("Unknown transformer: {}".format(cfg.transformer))
    