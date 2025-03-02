import torch
import torch.distributed as dist

import os
import random

# ----------------- Extra Components -----------------
from utils import distributed_utils
from utils.misc import MetricLogger, SmoothedValue
from utils.vis_tools import vis_data

# ----------------- Optimizer & LrScheduler Components -----------------
from utils.solver.optimizer import build_simple_optimizer, build_yolo_optimizer
from utils.solver.lr_scheduler import LinearWarmUpLrScheduler, build_lr_scheduler


class YoloTrainer(object):
    def __init__(self,
                 # Basic parameters
                 args,
                 cfg,
                 device,
                 # Model parameters
                 model,
                 model_ema,
                 criterion,
                 # Data parameters
                 train_loader,
                 evaluator,
                 ):
        # ------------------- basic parameters -------------------
        self.args = args
        self.cfg  = cfg
        self.epoch = 0
        self.best_map = -1.
        self.device = device
        self.criterion = criterion
        self.heavy_eval = False
        self.model_ema = model_ema
        # weak augmentatino stage
        self.second_stage = False
        self.second_stage_epoch = cfg.no_aug_epoch
        # path to save model
        self.path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
        os.makedirs(self.path_to_save, exist_ok=True)

        # ---------------------------- Dataset & Dataloader ----------------------------
        self.train_loader = train_loader

        # ---------------------------- Evaluator ----------------------------
        self.evaluator = evaluator

        # ---------------------------- Build Grad. Scaler ----------------------------
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

        # ---------------------------- Build Optimizer ----------------------------
        self.grad_accumulate = max(cfg.batch_size_base // args.batch_size, 1)
        cfg.base_lr = cfg.base_lr / cfg.batch_size_base * args.batch_size * self.grad_accumulate  # Auto scale learning rate
        cfg.min_lr  = cfg.base_lr * cfg.min_lr_ratio
        self.optimizer, self.start_epoch = build_yolo_optimizer(cfg, model, args.resume)

        # ---------------------------- Build LR Scheduler ----------------------------
        warmup_iters = cfg.warmup_epoch * len(self.train_loader)
        self.lr_scheduler_warmup = LinearWarmUpLrScheduler(warmup_iters, cfg.base_lr, cfg.warmup_bias_lr)
        self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer, args.resume)

        self.best_map = cfg.best_map / 100.0
        print("Best mAP metric: {}".format(self.best_map))

    def train(self, model):
        for epoch in range(self.start_epoch, self.cfg.max_epoch):
            if self.args.distributed:
                self.train_loader.batch_sampler.sampler.set_epoch(epoch)

            # check second stage
            if epoch >= (self.cfg.max_epoch - self.second_stage_epoch - 1) and not self.second_stage:
                self.check_second_stage()
                # save model of the last mosaic epoch
                weight_name = '{}_last_mosaic_epoch.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                print('Saving state of the last Mosaic epoch-{}.'.format(self.epoch))
                torch.save({'model': model.state_dict(),
                            'mAP': round(self.evaluator.map*100, 1),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)

            # train one epoch
            self.epoch = epoch
            self.train_one_epoch(model)

            # LR Schedule
            if (epoch + 1) > self.cfg.warmup_epoch:
                self.lr_scheduler.step()

            # eval one epoch
            if self.heavy_eval:
                model_eval = model.module if self.args.distributed else model
                self.eval(model_eval)
            else:
                model_eval = model.module if self.args.distributed else model
                if (epoch % self.cfg.eval_epoch) == 0 or (epoch == self.cfg.max_epoch - 1):
                    self.eval(model_eval)

            if self.args.debug:
                print("For debug mode, we only train 1 epoch")
                break

    def eval(self, model):
        # set eval mode
        model.eval()
        model_eval = model if self.model_ema is None else self.model_ema.ema
        cur_map = -1.
        to_save = False

        if distributed_utils.is_main_process():
            if self.evaluator is None:
                print('No evaluator ... save model and go on training.')
                to_save = True
                weight_name = '{}_no_eval.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
            else:
                print('Eval ...')
                # Evaluate
                with torch.no_grad():
                    self.evaluator.evaluate(model_eval)

                cur_map = self.evaluator.map
                if cur_map > self.best_map:
                    # update best-map
                    self.best_map = cur_map
                    to_save = True

            # Save model
            if to_save:
                print('Saving state, epoch:', self.epoch)
                weight_name = '{}_best.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                state_dicts = {
                    'model': model_eval.state_dict(),
                    'mAP': round(cur_map*100, 3),
                    'optimizer':  self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch': self.epoch,
                    'args': self.args,
                    }
                if self.model_ema is not None:
                    state_dicts["ema_updates"] = self.model_ema.updates
                torch.save(state_dicts, checkpoint_path)                      

        if self.args.distributed:
            # wait for all processes to synchronize
            dist.barrier()

        # set train mode.
        model.train()

    def train_one_epoch(self, model):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('size', SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('gnorm', SmoothedValue(window_size=1, fmt='{value:.1f}'))
        header = 'Epoch: [{} / {}]'.format(self.epoch, self.cfg.max_epoch)
        epoch_size = len(self.train_loader)
        print_freq = 100
        gnorm = 0.0

        # basic parameters
        epoch_size = len(self.train_loader)
        img_size   = self.cfg.train_img_size
        nw = epoch_size * self.cfg.warmup_epoch

        # Train one epoch
        for iter_i, (images, targets) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):
            ni = iter_i + self.epoch * epoch_size

            # Warmup
            if nw > 0 and ni < nw:
                self.lr_scheduler_warmup(ni, self.optimizer)
            elif ni == nw:
                print("Warmup stage is over.")
                self.lr_scheduler_warmup.set_lr(self.optimizer, self.cfg.base_lr)
                                
            # To device
            images = images.to(self.device, non_blocking=True).float()

            # Multi scale
            images, targets, img_size = self.rescale_image_targets(
                images, targets, self.cfg.max_stride, self.cfg.multi_scale)
                
            # Visualize train targets
            if self.args.vis_tgt:
                vis_data(images,
                         targets,
                         self.cfg.num_classes,
                         self.cfg.pixel_mean,
                         self.cfg.pixel_std,
                         )

            # Inference
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                outputs = model(images)
                # Compute loss
                loss_dict = self.criterion(outputs=outputs, targets=targets)
                losses = loss_dict['losses']
                losses /= self.grad_accumulate
                loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # Backward
            self.scaler.scale(losses).backward()

            # Optimize
            if (iter_i + 1) % self.grad_accumulate == 0:
                if self.cfg.clip_max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.cfg.clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # ModelEMA
                if self.model_ema is not None:
                    self.model_ema.update(model)

            # Update log
            metric_logger.update(**loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[2]["lr"])
            metric_logger.update(size=img_size)
            metric_logger.update(gnorm=gnorm)

            if self.args.debug:
                print("For debug mode, we only train 1 iteration")
                break

        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

    def rescale_image_targets(self, images, targets, max_stride, multi_scale_range=[0.5, 1.5]):
        """
            Deployed for Multi scale trick.
        """
        # During training phase, the shape of input image is square.
        old_img_size = images.shape[-1]
        min_img_size = old_img_size * multi_scale_range[0]
        max_img_size = old_img_size * multi_scale_range[1]

        # Choose a new image size
        new_img_size = random.randrange(min_img_size, max_img_size + max_stride, max_stride)
        
        # Resize
        if new_img_size != old_img_size:
            # interpolate
            images = torch.nn.functional.interpolate(
                                input=images, 
                                size=new_img_size, 
                                mode='bilinear', 
                                align_corners=False)
        # rescale targets
        for tgt in targets:
            boxes = tgt["boxes"].clone()
            labels = tgt["labels"].clone()
            boxes = torch.clamp(boxes, 0, old_img_size)
            # rescale box
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / old_img_size * new_img_size
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / old_img_size * new_img_size
            # refine tgt
            tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
            min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
            keep = (min_tgt_size >= 8)

            tgt["boxes"] = boxes[keep]
            tgt["labels"] = labels[keep]

        return images, targets, new_img_size

    def check_second_stage(self):
        # set second stage
        print('============== Second stage of Training ==============')
        self.second_stage = True
        self.heavy_eval = True

        # close mosaic augmentation
        if self.train_loader.dataset.mosaic_prob > 0.:
            print(' - Close < Mosaic Augmentation > ...')
            self.train_loader.dataset.mosaic_prob = 0.

        # close mixup augmentation
        if self.train_loader.dataset.mixup_prob > 0.:
            print(' - Close < Mixup Augmentation > ...')
            self.train_loader.dataset.mixup_prob = 0.

        # close copy-paste augmentation
        if self.train_loader.dataset.copy_paste > 0.:
            print(' - Close < Copy-paste Augmentation > ...')
            self.train_loader.dataset.copy_paste = 0.

class SimpleTrainer(object):
    def __init__(self,
                 # Basic parameters
                 args,
                 cfg,
                 device,
                 # Model parameters
                 model,
                 criterion,
                 # Data parameters
                 train_loader,
                 evaluator,
                 ):
        # ------------------- basic parameters -------------------
        self.args = args
        self.cfg  = cfg
        self.epoch = 0
        self.best_map = -1.
        self.device = device
        self.criterion = criterion

        # path to save model
        self.path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
        os.makedirs(self.path_to_save, exist_ok=True)

        # ---------------------------- Dataset & Dataloader ----------------------------
        self.train_loader = train_loader

        # ---------------------------- Evaluator ----------------------------
        self.evaluator = evaluator

        # ---------------------------- Build Optimizer ----------------------------
        self.grad_accumulate = max(cfg.batch_size_base // args.batch_size, 1)
        cfg.base_lr = cfg.base_lr / cfg.batch_size_base * args.batch_size * self.grad_accumulate  # Auto scale learning rate
        cfg.min_lr  = cfg.base_lr * cfg.min_lr_ratio
        self.optimizer, self.start_epoch = build_simple_optimizer(cfg, model, args.resume)

        # ---------------------------- Build LR Scheduler ----------------------------
        self.lr_scheduler_warmup = LinearWarmUpLrScheduler(cfg.warmup_iters, cfg.base_lr, cfg.warmup_bias_lr)
        self.lr_scheduler = build_lr_scheduler(cfg, self.optimizer, args.resume)

        self.best_map = cfg.best_map / 100.0
        print("Best mAP metric: {}".format(self.best_map))

    def train(self, model):
        for epoch in range(self.start_epoch, self.cfg.max_epoch):
            if self.args.distributed:
                self.train_loader.batch_sampler.sampler.set_epoch(epoch)

            # train one epoch
            self.epoch = epoch
            self.train_one_epoch(model)

            # LR Schedule
            self.lr_scheduler.step()

            # eval one epoch
            model_eval = model.module if self.args.distributed else model
            if (epoch % self.cfg.eval_epoch) == 0 or (epoch == self.cfg.max_epoch - 1):
                self.eval(model_eval)

            if self.args.debug:
                print("For debug mode, we only train 1 epoch")
                break

    def eval(self, model):
        # set eval mode
        model.eval()
        cur_map = -1.
        to_save = False

        if distributed_utils.is_main_process():
            if self.evaluator is None:
                print('No evaluator ... save model and go on training.')
                to_save = True
                weight_name = '{}_no_eval.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
            else:
                print('Eval ...')
                # Evaluate
                with torch.no_grad():
                    self.evaluator.evaluate(model)

                cur_map = self.evaluator.map
                if cur_map > self.best_map:
                    # update best-map
                    self.best_map = cur_map
                    to_save = True

            # Save model
            if to_save:
                print('Saving state, epoch:', self.epoch)
                weight_name = '{}_best.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                state_dicts = {
                    'model': model.state_dict(),
                    'mAP': round(cur_map*100, 3),
                    'optimizer':  self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch': self.epoch,
                    'args': self.args,
                    }
                torch.save(state_dicts, checkpoint_path)                      

        if self.args.distributed:
            # wait for all processes to synchronize
            dist.barrier()

        # set train mode.
        model.train()

    def train_one_epoch(self, model):
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('size', SmoothedValue(window_size=1, fmt='{value:d}'))
        metric_logger.add_meter('gnorm', SmoothedValue(window_size=1, fmt='{value:.1f}'))
        header = 'Epoch: [{} / {}]'.format(self.epoch, self.cfg.max_epoch)
        epoch_size = len(self.train_loader)
        print_freq = 100
        gnorm = 0.0

        # basic parameters
        epoch_size = len(self.train_loader)
        img_size   = self.cfg.train_img_size
        nw = self.cfg.warmup_iters

        # Train one epoch
        for iter_i, (images, targets) in enumerate(metric_logger.log_every(self.train_loader, print_freq, header)):
            ni = iter_i + self.epoch * epoch_size

            # Warmup
            if nw > 0 and ni < nw:
                self.lr_scheduler_warmup(ni, self.optimizer)
            elif ni == nw:
                print("Warmup stage is over.")
                self.lr_scheduler_warmup.set_lr(self.optimizer, self.cfg.base_lr)
                                
            # To device
            images = images.to(self.device, non_blocking=True).float()

            # Multi scale
            images, targets, img_size = self.rescale_image_targets(
                images, targets, 32, self.cfg.multi_scale)
                
            # Visualize train targets
            if self.args.vis_tgt:
                vis_data(images,
                         targets,
                         self.cfg.num_classes,
                         self.cfg.pixel_mean,
                         self.cfg.pixel_std,
                         )

            # Inference
            outputs = model(images)

            # Compute loss
            loss_dict = self.criterion(outputs=outputs, targets=targets)
            losses = loss_dict['losses']
            losses /= self.grad_accumulate
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # Backward
            losses.backward()

            # Optimize
            if (iter_i + 1) % self.grad_accumulate == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.cfg.clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update log
            metric_logger.update(**loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            metric_logger.update(size=img_size)
            metric_logger.update(gnorm=gnorm)

            if self.args.debug:
                print("For debug mode, we only train 1 iteration")
                break

        # Gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

    def rescale_image_targets(self, images, targets, max_stride, multi_scale_range=[0.5, 1.5]):
        """
            Deployed for Multi scale trick.
        """
        # During training phase, the shape of input image is square.
        old_img_size = images.shape[-1]
        min_img_size = old_img_size * multi_scale_range[0]
        max_img_size = old_img_size * multi_scale_range[1]

        # Choose a new image size
        new_img_size = random.randrange(min_img_size, max_img_size + max_stride, max_stride)
        
        # Resize
        if new_img_size != old_img_size:
            # interpolate
            images = torch.nn.functional.interpolate(
                                input=images, 
                                size=new_img_size, 
                                mode='bilinear', 
                                align_corners=False)
        # rescale targets
        for tgt in targets:
            boxes = tgt["boxes"].clone()
            labels = tgt["labels"].clone()
            boxes = torch.clamp(boxes, 0, old_img_size)
            # rescale box
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / old_img_size * new_img_size
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / old_img_size * new_img_size
            # refine tgt
            tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
            min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
            keep = (min_tgt_size >= 8)

            tgt["boxes"] = boxes[keep]
            tgt["labels"] = labels[keep]

        return images, targets, new_img_size
