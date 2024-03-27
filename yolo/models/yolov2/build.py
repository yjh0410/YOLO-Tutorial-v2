from .loss import SetCriterion
from .yolov2 import Yolov2


# build object detector
def build_yolov2(cfg, is_val=False):
    # -------------- Build YOLO --------------
    model = Yolov2(cfg, is_val)
  
    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)
        
    return model, criterion
