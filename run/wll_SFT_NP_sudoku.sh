#!/bin/bash
# Weighted likelihood using d1 likelihood calcuation + clippling + possibly pi_ref
export BASE_DATA="${BASE_DATA:-/home/rares/d1/data}"
echo "Saving to $BASE_DATA"

export VAR_DATA=$BASE_DATA/var_diff

export HF_DATASETS_CACHE=$BASE_DATA/cache_hugg
export HF_HOME=$BASE_DATA/cache_hugg
export HF_HUB_CACHE=$BASE_DATA/cache_hugg
export WANDB_DIR=$BASE_DATA/wandb

export LOGDIR=$BASE_DATA/var_diff/logs
mkdir -p $LOGDIR


export WANDB_PROJECT=var-diff-v2

MODEL_NAME=diffusion-reasoning/LLaDA-8B-Instruct-SFT
DATASET="sudoku"
RUN_NAME=wll_SFT_NP_${DATASET}
NUM_ITER=8 # number of policy gradient inner updates iterations
RL_RUN_NAME=${RUN_NAME}
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup accelerate launch \
    --config_file "wd1/accelerate.yaml" \
    --num_processes 4 \
    --main_process_port 12349 wd1/run_train.py \
    --config "wd1/train.yaml" \
    --model_path $MODEL_NAME \
    --num_iterations $NUM_ITER \
    --dataset $DATASET \
    --trainer_type wll_d1_neg \
    --run_name $RL_RUN_NAME \
    --wandb_project $WANDB_PROJECT \
    --output_dir $VAR_DATA/checkpoints/${RL_RUN_NAME} \
    --max_steps 5000 \
    > $LOGDIR/$RUN_NAME.log 2>&1 &