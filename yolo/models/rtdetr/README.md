# Real-time Transformer-based Object Detector:

## Results on the COCO-val
|     Model    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | ckpt | Logs |
|--------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|------|
| RT-DETR-R18  | 4xb4  |  640  |           45.5         |        63.5       |        66.8       |        21.0        | [ckpt](https://github.com/yjh0410/ODLab-World/releases/download/coco_weight/rtdetr_r18_coco.pth) | [log](https://github.com/yjh0410/ODLab-World/releases/download/coco_weight/RT-DETR-R18-COCO.txt)|
| RT-DETR-R50  | 4xb4  |  640  |           50.6         |        69.4       |       112.1       |        36.7        | [ckpt](https://github.com/yjh0410/ODLab-World/releases/download/coco_weight/rtdetr_r50_coco.pth) | [log](https://github.com/yjh0410/ODLab-World/releases/download/coco_weight/RT-DETR-R50-COCO.txt)|


## Train RT-DETR
### Single GPU
Taking training RT-DETR-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m rtdetr_r18 -bs 16  --fp16
```

### Multi GPU
Taking training RT-DETR on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root /data/datasets/ -m rtdetr_r18 -bs 16 --fp16 --sybn 
```

## Test RT-DETR
Taking testing RT-DETR on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m rtdetr_r18 --weight path/to/rtdetr_r18.pth --show 
```

## Evaluate RT-DETR
Taking evaluating RT-DETR on COCO-val as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m rtdetr_r18 -bs 16 --fp16 --resume path/to/rtdetr_r18.pth --eval_first
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m rtdetr_r18 --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m rtdetr_r18 --weight path/to/weight --show
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m rtdetr_r18 --weight path/to/weight --show
```
