# General Image Classification Laboratory


## Train a CNN
We have kindly provided a bash script `train.sh` to train the models. You can modify some hyperparameters in the script file according to your own needs.

For example, we are going to use 8 GPUs to train `ELANDarkNet-S` designed in this repo, so we can use the following command:

```Shell
bash train.sh elandarknet_s imagenet_1k path/to/imagnet_1k 8 1699 None
```

## Evaluate a CNN
- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on ImageNet-1K dataset:
```Shell
python main.py --cuda -dataset imagenet_1k --root path/to/imagnet_1k -m elandarknet_s --batch_size 256 --img_size 224 --eval --resume path/to/elandarknet_s.pth
```


## Experimental results
Tips:
- **Weak augmentation:** includes `random hflip` and `random crop resize`.
- **Strong augmentation:** includes `mixup`, `cutmix`, `rand aug`, `random erase` and so on. However, we don't use the strong augmentation.
- The `AdamW` with `weight decay = 0.05` and `base lr = 4e-3 (for bs of 4096)` is deployed as the optimzier, and the `CosineAnnealingLR` is deployed as the lr scheduler, where the `min lr` is set to 1e-6.

### ImageNet-1K
* Modified DarkNet (Yolov3's DarkNet with width and depth scaling factors)

|    Model      | Augment | Batch | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------|---------|-------|-------|------|-------|--------|--------|---------|
| DarkNet-S     |   weak  |  4096 |  100  | 224  |  68.5 |  1.6   |  4.6 M | [ckpt](https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/darknet_s_in1k_68.5.pth) |
| DarkNet-M     |   weak  |  4096 |  100  | 224  |       |        |        |  |
| DarkNet-L     |   weak  |  4096 |  100  | 224  |       |        |        |  |
| DarkNet-X     |   weak  |  4096 |  100  | 224  |       |        |        |  |

* Modified CSPDarkNet (Yolov5's DarkNet with width and depth scaling factors)

|    Model      | Augment | Batch | Epoch | size | acc@1 | GFLOPs | Params |  Weight |
|---------------|---------|-------|-------|------|-------|--------|--------|---------|
| CSPDarkNet-S  |   weak  |  4096 |  100  | 224  |  70.2 |  1.3   | 4.0 M  | [ckpt](https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/cspdarknet_s_in1k_70.2.pth) |
| CSPDarkNet-M  |   weak  |  4096 |  100  | 224  |       |        |        |  |
| CSPDarkNet-L  |   weak  |  4096 |  100  | 224  |       |        |        |  |
| CSPDarkNet-X  |   weak  |  4096 |  100  | 224  |       |        |        |  |

* ElANDarkNet (Yolov8's backbone)

|         Model          | Augment | Batch | Epoch | size | acc@1 | GFLOPs | Params  |  Weight |
|------------------------|---------|-------|-------|------|-------|--------|---------|---------|
| ElANDarkNet-N      |   weak  |  4096 |  100  | 224  |  62.1 |  0.38  | 1.36 M  | [ckpt](https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/elandarknet_n_in1k_62.1.pth) |
| ElANDarkNet-S      |   weak  |  4096 |  100  | 224  |  71.3 |  1.48  | 4.94 M  | [ckpt](https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/elandarknet_s_in1k_71.3.pth) |
| ElANDarkNet-M      |   weak  |  4096 |  100  | 224  |       |  4.67  | 11.60 M |  |
| ElANDarkNet-L      |   weak  |  4096 |  100  | 224  |       |  10.47 | 19.66 M |  |
| ElANDarkNet-X      |   weak  |  4096 |  100  | 224  |       |  20.56 | 37.86 M |  |


* GELAN (Proposed by YOLOv9)

|     Model     | Augment | Batch | Epoch | size | acc@1 | GFLOPs | Params  |  Weight |
|---------------|---------|-------|-------|------|-------|--------|---------|---------|
| GELAN-S       |   weak  |  4096 |  100  | 224  | 68.4  |  0.9   | 1.9 M   | [ckpt](https://github.com/yjh0410/ICLab/releases/download/in1k_pretrained/gelan_s_in1k_68.4.pth) |
| GELAN-C       |   weak  |  4096 |  100  | 224  |   |  5.2   | 8.8 M   | [ckpt]()|
