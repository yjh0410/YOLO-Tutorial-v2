from copy import deepcopy
import math
import torch
import torch.nn as nn


# ---------------------------- Model tools ----------------------------
def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

## Model EMA
class ModelEMA(object):
    def __init__(self, model, ema_decay=0.9999, ema_tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(self.de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: ema_decay * (1 - math.exp(-x / ema_tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def is_parallel(self, model):
        # Returns True if model is of type DP or DDP
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    def de_parallel(self, model):
        # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
        return model.module if self.is_parallel(model) else model

    def copy_attr(self, a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = self.de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        self.copy_attr(self.ema, model, include, exclude)
