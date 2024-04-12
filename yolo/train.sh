# Args parameters
MODEL=$1
DATASET=$2
DATASET_ROOT=$3
BATCH_SIZE=$4
WORLD_SIZE=$5
MASTER_PORT=$6
RESUME=$7


# -------------------------- Train Pipeline --------------------------
if [[ $WORLD_SIZE == 1 ]]; then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port ${MASTER_PORT} train.py \
            --cuda \
            --dataset ${DATASET} \
            --root ${DATASET_ROOT} \
            --model ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --resume ${RESUME} \
            --fp16
elif [[ $WORLD_SIZE -gt 1 && $WORLD_SIZE -le 8 ]]; then
    python -m torch.distributed.run --nproc_per_node=${WORLD_SIZE} --master_port ${MASTER_PORT} train.py \
            --cuda \
            --distributed \
            --dataset ${DATASET} \
            --root ${DATASET_ROOT} \
            --model ${MODEL} \
            --batch_size ${BATCH_SIZE} \
            --resume ${RESUME} \
            --fp16 \
            --sybn
else
    echo "The WORLD_SIZE is set to a value greater than 8, indicating the use of multi-machine \
          multi-card training mode, which is currently unsupported."
    exit 1
fi