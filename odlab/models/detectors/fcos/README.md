# FCOS: Fully Convolutional One-Stage Object Detector

Our `FCOS-R50-1x` baseline on COCO-val:
```Shell
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.579
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.428
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.685
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758
```

- FCOS

| Model        |  scale     |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| -------------| ---------- | ----- | ---------------------- |  ---------------  | ------ | ----- |
| FCOS_R18_1x  |  800,1333  |       |          34.1          |        52.2       | [ckpt](https://github.com/yjh0410/ODLab/releases/download/detection_weights/fcos_r18_1x_coco.pth) | [Logs](https://github.com/yjh0410/ODLab/releases/download/detection_weights/FCOS-R18-1x.txt) |
| FCOS_R50_1x  |  800,1333  |       |          39.1          |        57.9       | [ckpt](https://github.com/yjh0410/ODLab/releases/download/detection_weights/fcos_r50_1x_coco.pth) | [Logs](https://github.com/yjh0410/ODLab/releases/download/detection_weights/FCOS-R50-1x.txt) |

- Real-time FCOS

| Model          |  scale     |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| ---------------| ---------- | ----- | ---------------------- |  ---------------  | ------ | ----- |
| FCOS_RT_R18_4x |  512,736   |       |                        |                   |        |  |
| FCOS_RT_R50_4x |  512,736   |       |          43.9          |        60.2       |        |  |

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