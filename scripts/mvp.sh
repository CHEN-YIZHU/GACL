#!/bin/bash

MODE="mvp"
DATASET="cifar100" # cifar100, tinyimagenet, imagenet-r

N_TASKS=5
N=50
M=10
GPU_TRANSFORM="--gpu_transform"
USE_AMP="--use_amp"
GPU="--gpu"

if [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=500 ONLINE_ITER=3
    MODEL_NAME="mvp" EVAL_PERIOD=1000 
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="mvp" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
elif [ "$DATASET" == "imagenet-r" ]; then
    MEM_SIZE=0 ONLINE_ITER=3
    MODEL_NAME="mvp" EVAL_PERIOD=1000
    BATCHSIZE=64; LR=5e-3 OPT_NAME="adam" SCHED_NAME="default"
else
    echo "Undefined setting"
    exit 1
fi
NOTE="MVP"_"$MEM_SIZE" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)

for seed in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES="0" python main.py --mode $MODE \
    --dataset $DATASET \
    --n_tasks $N_TASKS --m $M --n $N \
    --rnd_seed $seed \
    --model_name $MODEL_NAME --opt_name $OPT_NAME --sched_name $SCHED_NAME \
    --lr $LR --batchsize $BATCHSIZE \
    --memory_size $MEM_SIZE $GPU_TRANSFORM --online_iter $ONLINE_ITER --data_dir local_datasets \
    --note $NOTE --eval_period $EVAL_PERIOD --n_worker 4 --rnd_NM \
    --use_mask --use_contrastiv --use_afs --use_gsf
done