# PlainDETR

Our `PlainDETR-R50-1x` baseline on COCO-val:
```Shell
```

## Results on COCO

| Model           |  Scale     |  Pretrained  |  FPS  | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | Weight | Logs  |
| --------------- | ---------- | ------------ | ----- | ---------------------- |  ---------------  | ------ | ----- |
| PlainDETR-R50   |  800,1333  |   IN1K-Cls   |       |                        |                   |  |  |
| PlainDETR-R50   |  800,1333  |   IN1K-MIM   |       |                        |                   |  |  |

- We explore whether PlainDETR can still be powerful when using ResNet as the backbone.
- We set up two comparative experiments, using the ResNet-50 pre-trained for the IN1K classification task and the ResNet-50 pre-trained by IN1K's MIM as the backbone of PlainDETR. Among them, we used the MIM pre-trained ResNet-50 provided by [SparK](https://github.com/keyu-tian/SparK).


## Train PlainDETR
### Single GPU
Taking training **PlainDETR** on COCO as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m plain_detr_r50 --batch_size 16 --eval_epoch 2
```

### Multi GPU
Taking training **PlainDETR** on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda -dist -d coco --root path/to/coco -m plain_detr_r50 --batch_size 16 --eval_epoch 2 
```

## Test PlainDETR
Taking testing **PlainDETR** on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m plain_detr_r50 --weight path/to/plain_detr_r50.pth -vt 0.4 --show 
```

## Evaluate PlainDETR
Taking evaluating **PlainDETR** on COCO-val as the example,
```Shell
python main.py --cuda -d coco --root path/to/coco -m plain_detr_r50 --resume path/to/plain_detr_r50.pth --eval_first
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m plain_detr_r50 --weight path/to/weight -vt 0.4 --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m plain_detr_r50 --weight path/to/weight -vt 0.4 --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m plain_detr_r50 --weight path/to/weight -vt 0.4 --show --gif
```