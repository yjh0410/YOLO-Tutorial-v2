from .loss import SetCriterion
from .rtdetr import RTDETR


# build object detector
def build_rtdetr(cfg, is_val=False):    
    # -------------- Build RT-DETR --------------
    model = RTDETR(cfg, is_val, use_nms=True, onnx_deploy=False)
            
    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)
        
    return model, criterion
