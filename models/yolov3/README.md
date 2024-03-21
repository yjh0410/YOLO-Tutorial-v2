# Redesigned YOLOv3:

- VOC

|   Model  | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|----------|-------|-------|-------------------|--------|--------|
| YOLOv3-S | 1xb16 |  640  |       75.5        | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v3/releases/download/yolo_tutorial_ckpt/yolov3_s_voc.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v3/releases/download/yolo_tutorial_ckpt/YOLOv3-S-VOC.txt) |

- COCO

|   Model  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |  Logs  |
|----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|--------|
| YOLOv3-S | 1xb16 |  640  |                    |               |   25.2            |   7.3             |  |  |

- For training, we train redesigned YOLOv3 with 150 epochs on COCO. We also gradient accumulate.
- For data augmentation, we only use the large scale jitter (LSJ), no Mosaic or Mixup augmentation.
- For optimizer, we use SGD with momentum 0.937, weight decay 0.0005 and base lr 0.01.
- For learning rate scheduler, we use linear decay scheduler.


## Train YOLOv3
### Single GPU
Taking training YOLOv3 on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov3 -bs 16 -size 640 --wp_epoch 3 --max_epoch 150 --eval_epoch 10 --no_aug_epoch 10 --ema --fp16 --multi_scale 
```

### Multi GPU
Taking training YOLOv3 on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root /data/datasets/ -m yolov3 -bs 128 -size 640 --wp_epoch 3 --max_epoch 150  --eval_epoch 10 --no_aug_epoch 20 --ema --fp16 --sybn --multi_scale --save_folder weights/ 
```

## Test YOLOv3
Taking testing YOLOv3 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov3 --weight path/to/yolov3.pth -size 640 -vt 0.3 --show 
```

## Evaluate YOLOv3
Taking evaluating YOLOv3 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco-val --root path/to/coco -m yolov3 --weight path/to/yolov3.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov3 --weight path/to/weight -size 640 -vt 0.3 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov3 --weight path/to/weight -size 640 -vt 0.3 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov3 --weight path/to/weight -size 640 -vt 0.3 --show --gif
```
