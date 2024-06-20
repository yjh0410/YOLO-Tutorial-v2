from .fcos import FcosPSS
from .criterion import SetCriterion


def build_fcos_pss(cfg, is_val=False):
    # ------------ build object detector ------------
    ## RT-FCOS    
    model = FcosPSS(cfg          = cfg,
                    conf_thresh  = cfg.train_conf_thresh if is_val else cfg.test_conf_thresh,
                    topk_results = cfg.train_topk        if is_val else cfg.test_topk,
                    )
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)

    return model, criterion
    