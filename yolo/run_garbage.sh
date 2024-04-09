

# -------------------------- Train Pipeline --------------------------
python -m torch.distributed.run --nproc_per_node=1 --master_port 9000 garbage.py --cuda
