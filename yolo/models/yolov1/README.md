# Redesigned YOLOv1:

- VOC

| Model  |  Backbone  | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|--------|------------|-------|-------|-------------------|--------|--------|
| YOLOv1 | ResNet-18  | 1xb16 |  640  |       75.0        | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/yolov1_r18_voc.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/YOLOv1-R18-VOC.txt) |

- COCO

| Model  |  Backbone  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight | Logs |
|--------|------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|------|
| YOLOv1 | ResNet-18  | 1xb16 |  640  |          27.6          |        46.8       |   37.8            |   21.3             | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/yolov1_r18_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/YOLOv1-R18-COCO.txt) |

- For training, we train redesigned YOLOv1 with 150 epochs on COCO.
- For data augmentation, we use the SSD's augmentation, including the RandomCrop, RandomDistort, RandomExpand, RandomHFlip and so on.
- For optimizer, we use AdamW with weight decay of 0.05 and per image base lr of 0.001 / 64.
- For learning rate scheduler, we use cosine decay scheduler.
- For batch size, we set it to 16, and we also use the gradient accumulation to approximate batch size of 256.


## Train YOLOv1
### Single GPU
Taking training YOLOv1-R18 on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov1_r18 -bs 16 --fp16 
```

### Multi GPU
Taking training YOLOv1-R18 on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root path/to/coco -m yolov1_r18 -bs 16 --fp16 
```

## Test YOLOv1
Taking testing YOLOv1-R18 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov1_r18 --weight path/to/yolov1.pth --show 
```

## Evaluate YOLOv1
Taking evaluating YOLOv1-R18 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov1_r18 --weight path/to/yolov1.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov1_r18 --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov1_r18 --weight path/to/weight --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov1_r18 --weight path/to/weight --show --gif
```
