#!/bin/bash

export BASE_DATA=/home/rares/d1/data
export VAR_DATA=$BASE_DATA/var_diff

export HF_DATASETS_CACHE=$BASE_DATA/cache_hugg
export HF_HOME=$BASE_DATA/cache_hugg
export HF_HUB_CACHE=$BASE_DATA/cache_hugg
export WANDB_DIR=$BASE_DATA/wandb

export LOGDIR=/home/rares/d1/data/var_diff/logs
export WANDB_PROJECT=var-diff-v2

# Configuration
MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct
GPU_IDS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_IDS[@]}
GPU_LIST=$(IFS=, ; echo "${GPU_IDS[*]}")
MASTER_PORT=29411

# Associative array for task → base checkpoint dir
declare -A BASE_CKPT_DIRS
BASE_CKPT_DIRS["gsm8k"]="${VAR_DATA}/checkpoints/wll_NP_gsm8k"
BASE_CKPT_DIRS["countdown"]="${VAR_DATA}/checkpoints/wll_NP_countdown"
BASE_CKPT_DIRS["math"]="${VAR_DATA}/checkpoints/wll_NP_math"
BASE_CKPT_DIRS["sudoku"]="${VAR_DATA}/checkpoints/wll_SFT_NP_sudoku"

# Add more if needed

# Associative array for task → checkpoint numbers (as strings)
declare -A CHECKPOINTS
CHECKPOINTS["gsm8k"]="1000 2500 5000 7500"
CHECKPOINTS["math"]="1000 2500 5000 7500"
CHECKPOINTS["countdown"]="1000 2500 4000"
CHECKPOINTS["sudoku"]="1000 2500 4000 5000"


# List of tasks and gen lengths
TASKS=("countdown" "sudoku" "gsm8k" "math")
GEN_LENGTHS=(512 256)

# Loop over tasks
for task in "${TASKS[@]}"; do
  ckpt_base="${BASE_CKPT_DIRS[$task]}"
  ckpt_list=${CHECKPOINTS[$task]}

  for ckpt_num in $ckpt_list; do
    CKPT_PATH="$ckpt_base/checkpoint-$ckpt_num"

    for gen_length in "${GEN_LENGTHS[@]}"; do
      MASTER_PORT=$(shuf -i 1000-2000 -n 1)
      echo "Using MASTER_PORT=$MASTER_PORT"
      # Batch size logic
      if [ "$gen_length" -eq 512 ]; then
        batch_size=8
      else
        batch_size=16
      fi

      echo "Evaluating $task @ checkpoint $ckpt_num, gen_length=$gen_length"

      CUDA_VISIBLE_DEVICES=$GPU_LIST python3 -m torch.distributed.run \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        eval/evalR.py \
        --dataset $task \
        --batch_size $batch_size \
        --gen_length $gen_length \
        --output_dir "${VAR_DATA}/eval_results_wll_np" \
        --model_path $MODEL_PATH \
        --checkpoint_path $CKPT_PATH
    done
  done
done

echo "All evaluations completed!"

python3 eval/parse_and_get_acc.py --directory "${VAR_DATA}/eval_results_wll_np"