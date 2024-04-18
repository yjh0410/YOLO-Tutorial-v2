# Redesigned YOLOv5:

- VOC

|   Model  | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|----------|-------|-------|-------------------|--------|--------|
| YOLOv5-S | 1xb16 |  640  |       79.0        | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/yolov5_s_voc.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/YOLOv5-S-VOC.txt) |

- COCO

|  Model   | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |  Logs  |
|----------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|--------|
| YOLOv5-S | 1xb16 |  640  |       38.8             |     56.9          |   27.3            |   9.0              | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/yolov5_s_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/YOLOv5-S-COCO.txt) |

- For training, we train redesigned YOLOv5 with 300 epochs on COCO. We also use the gradient accumulation.
- For data augmentation, we use the RandomAffine, RandomHSV, Mosaic and Mixup augmentation.
- For optimizer, we use AdamW with weight decay of 0.05 and per image base lr of 0.001 / 64.
- For learning rate scheduler, we use cosine decay scheduler.
- For batch size, we set it to 16, and we also use the gradient accumulation to approximate batch size of 256.


## Train YOLOv5
### Single GPU
Taking training YOLOv5-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov5_s -bs 16 --fp16 
```

### Multi GPU
Taking training YOLOv5-S on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root path/to/coco -m yolov5_s -bs 16 --fp16 
```

## Test YOLOv5
Taking testing YOLOv5-S on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov5_s --weight path/to/yolov5.pth --show 
```

## Evaluate YOLOv5
Taking evaluating YOLOv5-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov5_s --weight path/to/yolov5.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov5_s --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov5_s --weight path/to/weight --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov5_s --weight path/to/weight --show --gif
```
