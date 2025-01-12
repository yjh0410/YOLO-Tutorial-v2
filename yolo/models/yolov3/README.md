# Redesigned YOLOv3:

- VOC

|   Model  | Batch | Scale | AP<sup>val<br>0.5 | Weight |
|----------|-------|-------|-------------------|--------|
|  YOLOv3  | 1xb16 |  640  |               | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/yolov3_voc.pth) |

- COCO

|  Model  | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |
|---------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|
| YOLOv3  | 1xb16 |  640  |                    |               |               |                 | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/yolov3_coco.pth) |


## Train YOLOv3
### Single GPU
Taking training YOLOv3-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov3 -bs 16 --fp16 
```

### Multi GPU
Taking training YOLOv3-S on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root path/to/coco -m yolov3 -bs 16 --fp16 
```

## Test YOLOv3
Taking testing YOLOv3-S on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov3 --weight path/to/yolov3.pth --show 
```

## Evaluate YOLOv3
Taking evaluating YOLOv3-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov3 --weight path/to/yolov3.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov3 --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov3 --weight path/to/weight --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov3 --weight path/to/weight --show --gif
```
