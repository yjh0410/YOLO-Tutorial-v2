import torch.nn as nn

from .loss import SetCriterion
from .yolov7 import Yolov7


# build object detector
def build_yolov7(cfg, is_val=False):
    # -------------- Build YOLO --------------
    model = Yolov7(cfg, is_val)

    # -------------- Initialize YOLO --------------
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03    
            
    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)
        
    return model, criterion
