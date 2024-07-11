# RTCDet: My Empirical Study of Real-Time Convolutional Object Detectors.

- VOC

|     Model   | Batch | Scale | AP<sup>val<br>0.5 | Weight |  Logs  |
|-------------|-------|-------|-------------------|--------|--------|
| RTCDet-S    | 1xb16 |  640  |               |  |  |

- COCO

|    Model    | Batch | Scale | AP<sup>val<br>0.5:0.95 | AP<sup>val<br>0.5 | FLOPs<br><sup>(G) | Params<br><sup>(M) | Weight |  Logs  |
|-------------|-------|-------|------------------------|-------------------|-------------------|--------------------|--------|--------|
| RTCDet-S    | 1xb16 |  640  |                    |               |   26.9            |   8.9             |  |  |



## Train RTCDet
### Single GPU
Taking training RTCDet-S on COCO as the example,
```Shell
python train.py --cuda -d coco --root path/to/coco -m rtcdet_s -bs 16 --fp16 
```

### Multi GPU
Taking training RTCDet-S on COCO as the example,
```Shell
python -m torch.distributed.run --nproc_per_node=8 train.py --cuda --distributed -d coco --root path/to/coco -m rtcdet_s -bs 256 --fp16 
```

## Test RTCDet
Taking testing RTCDet-S on COCO-val as the example,
```Shell
python test.py --cuda -d coco --root path/to/coco -m rtcdet_s --weight path/to/RTCDet.pth --show 
```

## Evaluate RTCDet
Taking evaluating RTCDet-S on COCO-val as the example,
```Shell
python eval.py --cuda -d coco --root path/to/coco -m rtcdet_s --weight path/to/RTCDet.pth 
```

## Demo
### Detect with Image
```Shell
python demo.py --mode image --path_to_img path/to/image_dirs/ --cuda -m rtcdet_s --weight path/to/weight --show
```

### Detect with Video
```Shell
python demo.py --mode video --path_to_vid path/to/video --cuda -m rtcdet_s --weight path/to/weight --show --gif
```

### Detect with Camera
```Shell
python demo.py --mode camera --cuda -m rtcdet_s --weight path/to/weight --show --gif
```
