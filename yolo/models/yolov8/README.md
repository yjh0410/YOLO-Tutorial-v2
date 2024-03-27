# YOLOv8:

- VOC

|     Model   | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|-------------|-------|-------|-------------------|--------|--------|
| YOLOv8-S    | 1xb16 |  640  |               | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/yolov8_s_voc.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/YOLOv8-S-VOC.txt) |

- COCO

|    Model    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |  Logs  |
|-------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|--------|
| YOLOv8-S    | 1xb16 |  640  |                    |               |   26.9            |   8.9             |  |  |



## Train YOLOv8
### Single GPU
Taking training YOLOv8-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov8_s -bs 16 --fp16 
```

### Multi GPU
Taking training YOLOv8-S on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root path/to/coco -m yolov8_s -bs 256 --fp16 
```

## Test YOLOv8
Taking testing YOLOv8-S on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov8_s --weight path/to/yolov8.pth --show 
```

## Evaluate YOLOv8
Taking evaluating YOLOv8-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov8_s --weight path/to/yolov8.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov8_s --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov8_s --weight path/to/weight --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov8_s --weight path/to/weight --show --gif
```
