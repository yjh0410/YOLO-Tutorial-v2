# ------------------ Model Config ------------------
from .yolov1_config    import build_yolov1_config
from .yolov2_config    import build_yolov2_config
from .yolov3_config    import build_yolov3_config
from .yolov5_config    import build_yolov5_config
from .yolov5_af_config import build_yolov5af_config
from .yolov6_config    import build_yolov6_config
from .yolov8_config    import build_yolov8_config
from .gelan_config     import build_gelan_config
from .rtdetr_config    import build_rtdetr_config


def build_config(args):
    print('==============================')
    print('Model: {} ...'.format(args.model.upper()))
    # ----------- YOLO series -----------
    if   'yolov1' in args.model:
        cfg = build_yolov1_config(args)
    elif 'yolov2' in args.model:
        cfg = build_yolov2_config(args)
    elif 'yolov3' in args.model:
        cfg = build_yolov3_config(args)
    elif 'yolov5_af' in args.model:
        cfg = build_yolov5af_config(args)
    elif 'yolov5' in args.model:
        cfg = build_yolov5_config(args)
    elif 'yolov6' in args.model:
        cfg = build_yolov6_config(args)
    elif 'yolov8' in args.model:
        cfg = build_yolov8_config(args)
    elif 'gelan' in args.model:
        cfg = build_gelan_config(args)
    # ----------- RT-DETR -----------
    elif 'rtdetr' in args.model:
        cfg = build_rtdetr_config(args)

    else:
        raise NotImplementedError("Unknown model config: {}".format(args.model))
    
    # Print model config
    cfg.print_config()

    return cfg

