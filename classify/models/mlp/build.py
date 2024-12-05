from .mlp import MLP

def build_mlp(args):
    if args.model == "mlp":
        model = MLP(in_dim     = args.mlp_in_dim,
                    inter_dim  = 1024,
                    out_dim    = args.num_classes,
                    act_type   = "relu",
                    norm_type  = "bn")
        
    else:
        raise NotImplementedError("Unknown model: {}".format(args.model))
    
    return model
