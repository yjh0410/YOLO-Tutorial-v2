from .loss import SetCriterion
from .yolov3 import Yolov3


# build object detector
def build_yolov3(cfg, is_val=False):
    # -------------- Build YOLO --------------
    model = Yolov3(cfg, is_val)
  
    # -------------- Build criterion --------------
    criterion = None
    if is_val:
        # build criterion for training
        criterion = SetCriterion(cfg)
        
    return model, criterion
