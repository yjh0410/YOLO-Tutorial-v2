from .loss import SetCriterion
from .fcos import Fcos


# build object detector
def build_fcos(cfg, is_val=False):
    # -------------- Build YOLO --------------
    model = Fcos(cfg, is_val)
  
    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)
        
    return model, criterion
