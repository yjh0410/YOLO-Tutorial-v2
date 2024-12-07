# YOLOv9 (GElan):

- VOC

|     Model   | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|-------------|-------|-------|-------------------|--------|--------|
| YOLOv9-S    | 1xb16 |  640  |                   | [ckpt]() | [log]() |

- COCO

|    Model    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |  Logs  |
|-------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|--------|
| YOLOv9-S    | 1xb16 |  640  |                        |                   |   26.9            |   8.9             |  |  |



## Train YOLOv9
### Single GPU
Taking training YOLOv9-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov9_s -bs 16 --fp16 
```

### Multi GPU
Taking training YOLOv9-S on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root path/to/coco -m yolov9_s -bs 256 --fp16 
```

## Test YOLOv9
Taking testing YOLOv9-S on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov9_s --weight path/to/yolov9.pth --show 
```

## Evaluate YOLOv9
Taking evaluating YOLOv9-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov9_s --weight path/to/yolov9.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov9_s --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov9_s --weight path/to/weight --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov9_s --weight path/to/weight --show --gif
```
