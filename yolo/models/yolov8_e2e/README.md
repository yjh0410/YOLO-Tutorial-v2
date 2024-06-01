# End-to-End YOLOv8:

Inspired by YOLOv10, I deploy two parallel detection heads, one using one-to-many assinger (o2m head) and the other using one-to-one assinger (o2o head). To avoid conflicts between the gradients returned by o2o head and o2m head, we truncate the gradients returned from o2o head to the backbone and neck, and only allow the gradients returned from o2m head to update the backbone and neck. This operation is consistent with the practice of YOLOv10. For evaluation, we remove the o2m head and only use o2o head without NMS.

However, I have no GPU to train YOLOv8-E2E.

- VOC

|     Model   | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|-------------|-------|-------|-------------------|--------|--------|
| YOLOv8-E2E-S    | 1xb16 |  640  |               |  |  |

- COCO

|    Model    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |  Logs  |
|-------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|--------|
| YOLOv8-E2E-S    | 1xb16 |  640  |                    |               |   26.9            |   8.9             |  |  |



## Train YOLOv8-E2E
### Single GPU
Taking training YOLOv8-E2E-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m yolov8_e2e_s -bs 16 --fp16 
```

### Multi GPU
Taking training YOLOv8-E2E-S on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root path/to/coco -m yolov8_e2e_s -bs 256 --fp16 
```

## Test YOLOv8
Taking testing YOLOv8-E2E-S on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m yolov8_e2e_s --weight path/to/yolov8.pth --show 
```

## Evaluate YOLOv8
Taking evaluating YOLOv8-E2E-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m yolov8_e2e_s --weight path/to/yolov8.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m yolov8_e2e_s --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m yolov8_e2e_s --weight path/to/weight --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m yolov8_e2e_s --weight path/to/weight --show --gif
```
