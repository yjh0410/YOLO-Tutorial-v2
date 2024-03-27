# Redesigned YOLOv2:

- VOC

| Model  |  Backbone  | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|--------|------------|-------|-------|-------------------|--------|--------|
| YOLOv2 | ResNet-18  | 1xb16 |  640  |       75.7        | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/yolov2_r18_voc.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/YOLOv2-R18-VOC.txt) |

- COCO

| Model  |  Backbone  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|--------|------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv2 | ResNet-18  | 1xb16 |  640  |                    |               |   38.0            |   21.5             | [ckpt](https://github.com/yjh0410/RT-ODLab/releases/download/yolo_tutorial_ckpt/yolov2_coco.pth) |

- For training, we train redesigned YOLOv2 with 150 epochs on COCO.
- For data augmentation, we use the SSD's augmentation, including the RandomCrop, RandomDistort, RandomExpand, RandomHFlip and so on.
- For optimizer, we use AdamW with weight decay of 0.05 and per image base lr of 0.001 / 64.
- For learning rate scheduler, we use cosine decay scheduler.
- For batch size, we set it to 16, and we also use the gradient accumulation to approximate batch size of 256.


## Train YOLOv2
### Single GPU
Taking training YOLOv2-R18 on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov2_r18 -bs 16 --fp16 
```

### Multi GPU
Taking training YOLOv2-R18 on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root path/to/coco -m yolov2_r18 -bs 16 --fp16 
```

## Test YOLOv2
Taking testing YOLOv2-R18 on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov2_r18 --weight path/to/yolov2.pth --show 
```

## Evaluate YOLOv2
Taking evaluating YOLOv2-R18 on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov2_r18 --weight path/to/yolov2.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov2_r18 --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov2_r18 --weight path/to/weight --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov2_r18 --weight path/to/weight --show --gif
```
