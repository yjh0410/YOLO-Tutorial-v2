import os

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.custom_evaluator import CustomEvaluator



def build_evluator(args, cfg, transform, device):
    # Evaluator
    ## VOC Evaluator
    if args.dataset == 'voc':
        evaluator = VOCAPIEvaluator(cfg       = cfg,
                                    data_dir  = args.root,
                                    device    = device,
                                    transform = transform
                                    )
    ## COCO Evaluator
    elif args.dataset == 'coco':
        evaluator = COCOAPIEvaluator(cfg       = cfg,
                                     data_dir  = args.root,
                                     device    = device,
                                     transform = transform
                                     )
    ## Custom dataset Evaluator
    elif args.dataset == 'ourdataset':
        evaluator = CustomEvaluator(cfg       = cfg,
                                      data_dir  = args.root,
                                      device    = device,
                                      transform = transform
                                      )

    return evaluator
