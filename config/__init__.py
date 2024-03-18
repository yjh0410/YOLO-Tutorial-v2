# ------------------ Model Config ------------------
from .yolov1_config   import build_yolov1_config
from .yolov8_config   import build_yolov8_config
from .rtdetr_config import build_rtdetr_config

def build_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))
    # YOLOv8
    if 'yolov1' in args.model:
        cfg = build_yolov1_config(args)
    elif 'yolov8' in args.model:
        cfg = build_yolov8_config(args)
    # RT-DETR
    elif 'rtdetr' in args.model:
        cfg = build_rtdetr_config(args)
    else:
        raise NotImplementedError("Unknown model config: {}".format(args.model))
    
    # Print model config
    cfg.print_config()

    return cfg

