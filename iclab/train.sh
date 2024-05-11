# ------------------- Args setting -------------------
MODEL=$1
DATASET=$2
DATASET_ROOT=$3
WORLD_SIZE=$4
MASTER_PORT=$5
RESUME=$6

# ------------------- Training setting -------------------
## Epoch
BATCH_SIZE=128
GRAD_ACCUMULATE=32
WP_EPOCH=10
MAX_EPOCH=100
EVAL_EPOCH=5
DROP_PATH=0.1
## Scheduler
OPTIMIZER="adamw"
LRSCHEDULER="cosine"
BASE_LR=1e-3         # 0.1 for SGD; 0.001 for AdamW
MIN_LR=1e-6
BATCH_BASE=1024      # 256 for SGD; 1024 for AdamW
MOMENTUM=0.9
WEIGHT_DECAY=0.05    # 0.0001 for SGD; 0.05 for AdamW

# ------------------- Dataset config -------------------
if [[ $DATASET == "mnist" ]]; then
    IMG_SIZE=28
    NUM_CLASSES=10
elif [[ $DATASET == "cifar10" ]]; then
    IMG_SIZE=32
    NUM_CLASSES=10
elif [[ $DATASET == "cifar100" ]]; then
    IMG_SIZE=32
    NUM_CLASSES=100
elif [[ $DATASET == "imagenet_1k" || $DATASET == "imagenet_22k" ]]; then
    IMG_SIZE=224
    NUM_CLASSES=1000
elif [[ $DATASET == "custom" ]]; then
    IMG_SIZE=224
    NUM_CLASSES=2
else
    echo "Unknown dataset!!"
    exit 1
fi


# ------------------- Training pipeline -------------------
if [ $WORLD_SIZE == 1 ]; then
    python train.py \
            --cuda \
            --root ${DATASET_ROOT} \
            --dataset ${DATASET} \
            --model ${MODEL} \
            --resume ${RESUME} \
            --batch_size ${BATCH_SIZE} \
            --batch_base ${BATCH_BASE} \
            --grad_accumulate ${GRAD_ACCUMULATE} \
            --img_size ${IMG_SIZE} \
            --drop_path ${DROP_PATH} \
            --max_epoch ${MAX_EPOCH} \
            --wp_epoch ${WP_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --optimizer ${OPTIMIZER} \
            --lr_scheduler ${LRSCHEDULER} \
            --base_lr ${BASE_LR} \
            --min_lr ${MIN_LR} \
            --momentum ${MOMENTUM} \
            --weight_decay ${WEIGHT_DECAY} \
            --color_jitter 0.0 \
            --reprob 0.0 \
            --mixup 0.0 \
            --cutmix 0.0

elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port ${MASTER_PORT} train.py \
            --cuda \
            --distributed \
            --root ${DATASET_ROOT} \
            --dataset ${DATASET} \
            --model ${MODEL} \
            --resume ${RESUME} \
            --batch_size ${BATCH_SIZE} \
            --batch_base ${BATCH_BASE} \
            --grad_accumulate ${GRAD_ACCUMULATE} \
            --img_size ${IMG_SIZE} \
            --drop_path ${DROP_PATH} \
            --max_epoch ${MAX_EPOCH} \
            --wp_epoch ${WP_EPOCH} \
            --eval_epoch ${EVAL_EPOCH} \
            --optimizer ${OPTIMIZER} \
            --lr_scheduler ${LRSCHEDULER} \
            --base_lr ${BASE_LR} \
            --min_lr ${MIN_LR} \
            --momentum ${MOMENTUM} \
            --weight_decay ${WEIGHT_DECAY} \
            --sybn \
            --color_jitter 0.0 \
            --reprob 0.0 \
            --mixup 0.0 \
            --cutmix 0.0
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi


# # ------------------- Training pipeline with strong augmentations -------------------
# if [ $WORLD_SIZE == 1 ]; then
#     python train.py \
#             --cuda \
#             --root ${DATASET_ROOT} \
#             --dataset ${DATASET} \
#             --model ${MODEL} \
#             --resume ${RESUME} \
#             --batch_size ${BATCH_SIZE} \
#             --batch_base ${BATCH_BASE} \
#             --grad_accumulate ${GRAD_ACCUMULATE} \
#             --img_size ${IMG_SIZE} \
#             --drop_path ${DROP_PATH} \
#             --max_epoch ${MAX_EPOCH} \
#             --wp_epoch ${WP_EPOCH} \
#             --eval_epoch ${EVAL_EPOCH} \
#             --optimizer ${OPTIMIZER} \
#             --lr_scheduler ${LRSCHEDULER} \
#             --base_lr ${BASE_LR} \
#             --min_lr ${MIN_LR} \
#             --weight_decay ${WEIGHT_DECAY} \
#             --aa "rand-m9-mstd0.5-inc1" \
#             --reprob 0.25 \
#             --mixup 0.8 \
#             --cutmix 1.0
# elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
#     python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port ${MASTER_PORT} train.py \
#             --cuda \
#             --distributed \
#             --root ${DATASET_ROOT} \
#             --dataset ${DATASET} \
#             --model ${MODEL} \
#             --resume ${RESUME} \
#             --batch_size ${BATCH_SIZE} \
#             --batch_base ${BATCH_BASE} \
#             --grad_accumulate ${GRAD_ACCUMULATE} \
#             --img_size ${IMG_SIZE} \
#             --drop_path ${DROP_PATH} \
#             --max_epoch ${MAX_EPOCH} \
#             --wp_epoch ${WP_EPOCH} \
#             --eval_epoch ${EVAL_EPOCH} \
#             --optimizer ${OPTIMIZER} \
#             --lr_scheduler ${LRSCHEDULER} \
#             --base_lr ${BASE_LR} \
#             --min_lr ${MIN_LR} \
#             --weight_decay ${WEIGHT_DECAY} \
#             --sybn \
#             --aa "rand-m9-mstd0.5-inc1" \
#             --reprob 0.25 \
#             --mixup 0.8 \
#             --cutmix 1.0
# else
#     echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
#           multi-card training mode, which is currently unsupported."
#     exit 1
# fi
