# FCOS: Fully Convolutional One-Stage Object Detector


- COCO

| Model          |  scale     |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| ---------------| ---------- | ----- | ---------------------- |  ---------------  | ------ | ----- |
| FCOS_R18_1x    |  800,1333  |       |                    |               | [ckpt]() | [Logs]() |
| FCOS_R50_1x    |  800,1333  |       |                        |                   | [ckpt]() | [Logs]() |
| FCOS_RT_R18_4x |  512,736   |       |                        |                   | [ckpt]() | [Logs]() |
| FCOS_RT_R50_4x |  512,736   |       |                        |                   | [ckpt]() | [Logs]() |

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