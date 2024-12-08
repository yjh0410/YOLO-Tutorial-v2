from .loss import SetCriterion
from .yolof import Yolof


# build object detector
def build_yolof(cfg, is_val=False):
    # -------------- Build YOLO --------------
    model = Yolof(cfg, is_val)
  
    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)
        
    return model, criterion
