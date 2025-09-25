#!/bin/bash

export BASE_DATA="${BASE_DATA:-data}"
echo "Saving to $BASE_DATA"

export VAR_DATA=$BASE_DATA/var_diff
export HF_DATASETS_CACHE=$BASE_DATA/cache_hugg
export HF_HOME=$BASE_DATA/cache_hugg
export HF_HUB_CACHE=$BASE_DATA/cache_hugg
export WANDB_DIR=$BASE_DATA/wandb
export LOGDIR=$BASE_DATA/var_diff/logs
if [ ! -d $LOGDIR ]; then
    mkdir -p $LOGDIR
else
    echo "log directory exists"
fi

export WANDB_PROJECT=var-diff-v2
MODEL_NAME=GSAI-ML/LLaDA-8B-Instruct

############## Input arguments ##################

# number of gpus
NUM_GPUS=$1 # [8, 4, 1]

# per-device batch size
PD_BATCH_SIZE=$2 # [1, 4]; if reg is activated (beta>0) set to 1, else 4 for efficiency

# objective for training
LOSS_TYPE=$3 # [grpo, wd1]; grpo is mdpo here

# dataset: 
DATA=$4 # [numina, math]

############## End of input arguments ##################

DIFFU_STEPS=256
GRAD_ACCUM=$((8 / PD_BATCH_SIZE))  # 16 / per_device_batch_size
# DATASET=open-r1/OpenR1-Math-220k
# DATASET=ankner/math-500
if [[ $DATA == "math" ]]; then
    DATASET=ankner/math-500
elif [[ $DATA == "gsm8k" ]]; then
    DATASET=openai/gsm8k
elif [[ $DATA == "countdown" ]]; then
    DATASET=countdown
elif [[ $DATA == "sudoku" ]]; then
    DATASET=sudoku
else
    DATASET=open-r1/OpenR1-Math-220k
fi

OUTPUT_DIR=checkpoints/LLaDA-8B-Instruct-MDPO-${LOSS_TYPE}-${DATA}-adv-${DIFFU_STEPS}st-8sample_temp0.4_${NUM_GPUS}gpus

# CUDA_VISIBLE_DEVICES=0,1,2,3 
accelerate launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes $NUM_GPUS src/open_r1/mdpo.py  \
    --model_name_or_path $MODEL_NAME \
    --config recipes/LLaDA-Instruct/mdpo/config_demo.yaml  \
    --dataset_train_split train  \
    --rl_loss_type $LOSS_TYPE\
    --num_train_epochs 1  \
    --dataset_name $DATASET \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 20 \
    --output_dir $OUTPUT_DIR \
    --num_generations $NUM_GPUS \
    --per_device_train_batch_size $PD_BATCH_SIZE \
    --learning_rate 1e-6  \
    --warmup_ratio 0.0 \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --temperature 0.4  \
    --beta 0.0  \
    --block_length 128  \
    --max_completion_length 512  \
    --sample_train_steps 8  \
    --max_prompt_length 320  \
    --diffusion_steps $DIFFU_STEPS  \
    --remask_strategy low_confidence  \
    --overtime_conf true \
    --num_train_samples 7400  \
    --system_prompt "Let's think step by step and output the final answer within \\boxed{}." \
    --incremental_training false \
    --mixture_data true \
    --eval_on_start false \
    --max_steps 150  # Stop after 150 steps
    # > $LOGDIR/$RUN_NAME.log 2>&1 #&
    # --ab_path ab_samples/ab_from_40k_epoch_1.csv
    # --lr_scheduler_type linear \
