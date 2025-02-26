# FCOS: Fully Convolutional One-Stage Object Detector


- COCO

| Model          |  scale     |  FPS<sup>FP32<br>RTX 4060  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| ---------------| ---------- | -------------------------- | ---------------------- |  ---------------  | ------ | ----- |
| FCOS_R18_1x    |  800,1333  |            24              |          34.0          |        52.2       | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/fcos_r18_1x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/FCOS-R18-1x.txt) |
| FCOS_R50_1x    |  800,1333  |             9              |          39.0          |        58.3       | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/fcos_r50_1x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/FCOS-R50-1x.txt) |
| FCOS_RT_R18_3x |  512,736   |            56              |          35.8          |        53.3       | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/fcos_rt_r18_3x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/FCOS-RT-R18-3x.txt) |
| FCOS_RT_R50_3x |  512,736   |            34              |          40.7          |        59.3       | [ckpt](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/fcos_rt_r50_3x_coco.pth) | [log](https://github.com/yjh0410/YOLO-Tutorial-v2/releases/download/yolo_tutorial_ckpt/FCOS-RT-R50-3x.txt) |

## Train FCOS
### Single GPU
Taking training **FCOS_R18_1x** on COCO as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m fcos_r18_1x --batch_size 16 --eval_epoch 2
```

### Multi GPU
Taking training **FCOS_R18_1x** on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root path/to/coco -m fcos_r18_1x --batch_size 16 --eval_epoch 2 
```

## Test FCOS
Taking testing **FCOS_R18_1x** on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m fcos_r18_1x --weight path/to/fcos_r18_1x.pth -vt 0.4 --show 
```

## Evaluate FCOS
Taking evaluating **FCOS_R18_1x** on COCO-val as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m fcos_r18_1x --resume path/to/fcos_r18_1x.pth --eval_first
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m fcos_r18_1x --weight path/to/weight -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m fcos_r18_1x --weight path/to/weight -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m fcos_r18_1x --weight path/to/weight -vt 0.4 --show --gif
```