from .loss import SetCriterion
from .yolov4 import Yolov4


# build object detector
def build_yolov4(cfg, is_val=False):
    # -------------- Build YOLO --------------
    model = Yolov4(cfg, is_val)
  
    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)
        
    return model, criterion
