# YOLO11:

- VOC

|     Model   | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|-------------|-------|-------|-------------------|--------|--------|
| YOLO11-S    | 1xb16 |  640  |      83.6     | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/yolo11_s_voc.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v5/releases/download/yolo_tutorial_ckpt/YOLO11-S-VOC.txt) |

- COCO

|    Model    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |  Logs  |
|-------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|--------|
| YOLO11-S    | 1xb16 |  640  |                    |               |   26.9            |   8.9             |  |  |



## Train YOLO11
### Single GPU
Taking training YOLO11-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolo11_s -bs 16 --fp16 
```

### Multi GPU
Taking training YOLO11-S on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root path/to/coco -m yolo11_s -bs 256 --fp16 
```

## Test YOLO11
Taking testing YOLO11-S on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolo11_s --weight path/to/yolo11.pth --show 
```

## Evaluate YOLO11
Taking evaluating YOLO11-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolo11_s --weight path/to/yolo11.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolo11_s --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolo11_s --weight path/to/weight --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolo11_s --weight path/to/weight --show --gif
```
