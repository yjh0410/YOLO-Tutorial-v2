from .yolof_head     import YolofHead
from .fcos_head      import FcosHead, FcosRTHead, FcosPSSHead


# build head
def build_head(cfg, in_dim, out_dim):
    print('==============================')
    print('Head: {}'.format(cfg.head))
    
    if   cfg.head == 'fcos_head':
        model = FcosHead(cfg, in_dim, out_dim)
    elif cfg.head == 'fcos_rt_head':
        model = FcosRTHead(cfg, in_dim, out_dim)
    elif cfg.head == 'fcos_pss_head':
        model = FcosPSSHead(cfg, in_dim, out_dim)
    elif cfg.head == 'yolof_head':
        model = YolofHead(cfg, in_dim, out_dim)

    return model
