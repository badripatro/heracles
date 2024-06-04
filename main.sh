#!/bin/bash

echo "Current path is $PATH"
echo "Running"
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# heracles_s
python3 -m torch.distributed.launch \
   --nproc_per_node=8 \
   --nnodes=1 \
   --node_rank=0 \
   --master_addr="localhost" \
   --master_port=12346 \
   --use_env main.py --config configs/heracles/heracles_s.py --data-path ../../../../dataset/Image_net/imagenet --epochs 310 --batch-size 128 \
   --token-label --token-label-size 7 --token-label-data ../../../../dataset/Image_net/imagenet_efficientnet_l2_sz475_top5/


# # heracles_b
# python3 -m torch.distributed.launch \
#    --nproc_per_node=8 \
#    --nnodes=1 \
#    --nnodes=1 \
#    --node_rank=0 \
#    --master_addr="localhost" \
#    --master_port=12346 \
#    --use_env main.py --config configs/heracles/heracles_b.py --data-path ../../../../dataset/Image_net/imagenet --epochs 310 --batch-size 128 \
#    --token-label --token-label-size 7 --token-label-data ../../../../dataset/Image_net/imagenet_efficientnet_l2_sz475_top5/

# # heracles_l
# python3 -m torch.distributed.launch \
#    --nproc_per_node=8 \
#    --nnodes=1 \
#    --node_rank=0 \
#    --master_addr="localhost" \
#    --master_port=12346 \
#    --use_env main.py --config configs/heracles/heracles_l.py --data-path ../../../../dataset/Image_net/imagenet --epochs 310 --batch-size 128 \
#    --token-label --token-label-size 7 --token-label-data ../../../../dataset/Image_net/imagenet_efficientnet_l2_sz475_top5/