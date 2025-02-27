# ------------------ Model Config ------------------
from .yolov1_config  import build_yolov1_config
from .yolov2_config  import build_yolov2_config
from .yolov3_config  import build_yolov3_config
from .yolov4_config  import build_yolov4_config
from .yolov5_config  import build_yolov5_config
from .yolox_config   import build_yolox_config
from .yolov6_config  import build_yolov6_config
from .yolov7_config  import build_yolov7_config
from .yolov8_config  import build_yolov8_config
from .yolov9_config  import build_yolov9_config
from .yolov10_config  import build_yolov10_config
from .yolo11_config  import build_yolo11_config

from .yolof_config   import build_yolof_config
from .fcos_config    import build_fcos_config
from .rtdetr_config  import build_rtdetr_config


def build_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))
    # ----------- YOLO series -----------
    if   'yolov10' in args.model:
        cfg = build_yolov10_config(args)
    elif 'yolov1' in args.model:
        cfg = build_yolov1_config(args)
    elif 'yolov2' in args.model:
        cfg = build_yolov2_config(args)
    elif 'yolov3' in args.model:
        cfg = build_yolov3_config(args)
    elif 'yolov4' in args.model:
        cfg = build_yolov4_config(args)
    elif 'yolox' in args.model:
        cfg = build_yolox_config(args)
    elif 'yolov5' in args.model:
        cfg = build_yolov5_config(args)
    elif 'yolov6' in args.model:
        cfg = build_yolov6_config(args)
    elif 'yolov7' in args.model:
        cfg = build_yolov7_config(args)
    elif 'yolov8' in args.model:
        cfg = build_yolov8_config(args)
    elif 'yolov9' in args.model:
        cfg = build_yolov9_config(args)
    elif 'yolov10' in args.model:
        cfg = build_yolov10_config(args)
    elif 'yolo11' in args.model:
        cfg = build_yolo11_config(args)
        
    # ----------- RT-DETR -----------
    elif 'yolof' in args.model:
        cfg = build_yolof_config(args)
    elif 'fcos' in args.model:
        cfg = build_fcos_config(args)
    elif 'rtdetr' in args.model:
        cfg = build_rtdetr_config(args)

    else:
        raise NotImplementedError("Unknown model config: {}".format(args.model))
    
    # Print model config
    cfg.print_config()

    return cfg

