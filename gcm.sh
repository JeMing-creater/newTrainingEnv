export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun \
    --nproc_per_node 8 \
    --master_port 25000 \
    trainer.py jsons/training_config_gcm.json