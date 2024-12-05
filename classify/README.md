# General Image Classification Laboratory

## Train
For example, we are going to train `ConvNet` designed in this repo, so we can use the following command:

```Shell
cd Vision-Pretraining-Tutorial/image_classification/
python main.py --cuda \
               --dataset cifar \
               --model convnet \
               --batch_size 256 \
               --optimizer adamw \
               --base_lr 1e-3 \
               --min_lr 1e-6
```

## Evaluate
- Evaluate the `top1 & top5` accuracy:
```Shell
cd Vision-Pretraining-Tutorial/image_classification/
python main.py --cuda \
               --dataset cifar \
               --model convnet \
               --batch_size 256 \
               --eval \
               --resume path/to/checkpoint
```