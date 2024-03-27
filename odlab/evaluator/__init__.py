from evaluator.coco_evaluator import COCOAPIEvaluator


def build_evluator(args, cfg, device, testset=False):
    evaluator = None
    # COCO Evaluator
    if args.dataset == 'coco':
        evaluator = COCOAPIEvaluator(args, cfg, device, testset)

    return evaluator
