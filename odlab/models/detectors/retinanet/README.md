# RetinaNet

Our `RetinaNet-R50-1x` baseline on COCO-val:
```Shell

```

- ImageNet-1K_V1 pretrained

| Model             |  scale     |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| ------------------| ---------- | ----- | ---------------------- |  ---------------  | ------ | ----- |
| RetinaNet_R18_1x  |  800,1333  |       |          30.5          |        48.1       | [ckpt](https://github.com/yjh0410/ODLab/releases/download/detection_weights/retinanet_r18_1x_coco.pth) | [log](https://github.com/yjh0410/ODLab/releases/download/detection_weights/RetinaNet-R18-1x.txt) |
| RetinaNet_R50_1x  |  800,1333  |       |                        |                   |  |  |


## Train RetinaNet
### Single GPU
Taking training **RetinaNet_R18_1x** on COCO as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m retinanet_r18_1x --batch_size 16 --eval_epoch 2
```

### Multi GPU
Taking training **RetinaNet_R18_1x** on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root path/to/coco -m retinanet_r18_1x --batch_size 16 --eval_epoch 2 
```

## Test RetinaNet
Taking testing **RetinaNet_R18_1x** on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m retinanet_r18_1x --weight path/to/retinanet_r18_1x.pth -vt 0.4 --show 
```

## Evaluate RetinaNet
Taking evaluating **RetinaNet_R18_1x** on COCO-val as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m retinanet_r18_1x --resume path/to/retinanet_r18_1x.pth --eval_first
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m retinanet_r18_1x --weight path/to/weight -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m retinanet_r18_1x --weight path/to/weight -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m retinanet_r18_1x --weight path/to/weight -vt 0.4 --show --gif
```