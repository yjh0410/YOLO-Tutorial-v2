import torch.nn as nn

from .loss import SetCriterion
from .rtcdet import RTCDet


# build object detector
def build_rtcdet(cfg, is_val=False):
    # -------------- Build YOLO --------------
    model = RTCDet(cfg, is_val)

    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)
        
    return model, criterion
