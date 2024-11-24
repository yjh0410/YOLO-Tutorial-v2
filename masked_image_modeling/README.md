# Masked AutoEncoder

## 1. Pretrain
We have kindly provided the bash script `main_pretrain.sh` file for pretraining. You can modify some hyperparameters in the script file according to your own needs.

```Shell
cd Vision-Pretraining-Tutorial/masked_image_modeling/
python main_pretrain.py --cuda \
                        --dataset cifar10 \
                        --model vit_t \
                        --mask_ratio 0.75 \
                        --batch_size 128 \
                        --optimizer adamw \
                        --weight_decay 0.05 \
                        --lr_scheduler cosine \
                        --base_lr 0.00015 \
                        --min_lr 0.0 \
                        --max_epoch 400 \
                        --eval_epoch 20
```

## 2. Finetune
We have kindly provided the bash script `main_finetune.sh` file for finetuning. You can modify some hyperparameters in the script file according to your own needs.

```Shell
cd Vision-Pretraining-Tutorial/masked_image_modeling/
python main_finetune.py --cuda \
                        --dataset cifar10 \
                        --model vit_t \
                        --batch_size 256 \
                        --optimizer adamw \
                        --weight_decay 0.05 \
                        --base_lr 0.0005 \
                        --min_lr 0.000001 \
                        --max_epoch 100 \
                        --wp_epoch 5 \
                        --eval_epoch 5 \
                        --pretrained path/to/vit_t.pth
```
## 3. Evaluate 
- Evaluate the `top1 & top5` accuracy of `ViT-Tiny` on CIFAR10 dataset:
```Shell
python main_finetune.py --cuda \
                        --dataset cifar10 \
                        -m vit_t \
                        --batch_size 256 \
                        --eval \
                        --resume path/to/vit_t_cifar10.pth
```


## 4. Visualize Image Reconstruction
- Evaluate `ViT-Tiny` pretrained by MAE framework on CIFAR10 dataset:
```Shell
python main_pretrain.py --cuda \
                        --dataset cifar10 \
                        -m vit_t \
                        --resume path/to/mae_vit_t_cifar10.pth \
                        --eval \
                        --batch_size 1
```


## 5. Experiments
- On CIFAR10

| Method |  Model  | Epoch | Top 1    | Weight |  MAE weight  |
|  :---: |  :---:  | :---: | :---:    | :---:  |    :---:     |
|  MAE   |  ViT-T  | 100   |   91.2   | [ckpt](https://github.com/yjh0410/MAE/releases/download/checkpoints/ViT-T_Cifar10.pth) | [ckpt](https://github.com/yjh0410/MAE/releases/download/checkpoints/MAE_ViT-T_Cifar10.pth) |


## 6. Acknowledgment
Thank you to **Kaiming He** for his inspiring work on [MAE](http://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf). His research effectively elucidates the semantic distinctions between vision and language, offering valuable insights for subsequent vision-related studies. I would also like to express my gratitude for the official source code of [MAE](https://github.com/facebookresearch/mae). Additionally, I appreciate the efforts of [**IcarusWizard**](https://github.com/IcarusWizard) for reproducing the [MAE](https://github.com/IcarusWizard/MAE) implementation.
