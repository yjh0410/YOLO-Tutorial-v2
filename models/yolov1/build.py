from .loss import SetCriterion
from .yolov1 import Yolov1


# build object detector
def build_yolov1(cfg, is_val=False):
    # -------------- Build YOLO --------------
    model = Yolov1(cfg, is_val)
  
    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)
        
    return model, criterion
