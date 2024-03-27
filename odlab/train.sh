# Args setting
MODEL=$1
DATASET=$2
DATA_ROOT=$3
WORLD_SIZE=$4
MASTER_PORT=$5
if [[ $MODEL == *"yolof"* ]]; then
    # Epoch setting
    BATCH_SIZE=64
    EVAL_EPOCH=2
elif [[ $MODEL == *"fcos"* ]]; then
    # Epoch setting
    BATCH_SIZE=16
    EVAL_EPOCH=2
elif [[ $MODEL == *"retinanet"* ]]; then
    # Epoch setting
    BATCH_SIZE=16
    EVAL_EPOCH=2
elif [[ $MODEL == *"plain_detr"* ]]; then
    # Epoch setting
    BATCH_SIZE=16
    EVAL_EPOCH=2
elif [[ $MODEL == *"rtdetr"* ]]; then
    # Epoch setting
    BATCH_SIZE=16
    EVAL_EPOCH=1
fi

# -------------------------- Train Pipeline --------------------------
if [ $WORLD_SIZE == 1 ]; then
    python main.py \
        --cuda \
        --dataset ${DATASET}  \
        --root ${DATA_ROOT} \
        --model ${MODEL} \
        --batch_size ${BATCH_SIZE} \
        --eval_epoch ${EVAL_EPOCH}
elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=$WORLD_SIZE --master_port ${MASTER_PORT}  \
        main.py \
        --cuda \
        --distributed \
        --dataset ${DATASET}  \
        --root ${DATA_ROOT} \
        --model ${MODEL} \
        --batch_size ${BATCH_SIZE} \
        --eval_epoch ${EVAL_EPOCH}
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi