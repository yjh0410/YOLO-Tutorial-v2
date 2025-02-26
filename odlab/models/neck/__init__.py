from .dilated_encoder import DilatedEncoder
from .fpn import BasicFPN
from typing import List

# build neck
def build_neck(cfg, in_dim, out_dim):
    print('==============================')
    print('Neck: {}'.format(cfg.neck))

    # ----------------------- Neck module -----------------------
    if cfg.neck == 'dilated_encoder':
        model = DilatedEncoder(cfg, in_dim, out_dim)
        
    # ----------------------- FPN Neck -----------------------
    elif cfg.neck == 'basic_fpn':
        assert isinstance(in_dim, List)
        model = BasicFPN(cfg, in_dim, out_dim)
    else:
        raise NotImplementedError("Unknown Neck: <{}>".format(cfg.fpn))
        
    return model
