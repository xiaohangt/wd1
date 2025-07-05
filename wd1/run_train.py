import torch
import torch.distributed as dist
import wandb
from data_utils import (
    get_countdown_questions,
    get_gsm8k_questions,
    get_math_questions,
    get_sudoku_questions,
    set_random_seed,
)
from peft import LoraConfig
from reward_func import (
    boxed_and_answer_tags_format_reward,
    correctness_reward_func,
    correctness_reward_func_math,
    countdown_reward_func,
    int_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    sudoku_reward_func,
    xmlcount_reward_func,
)

# Custom imports
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from trl import ModelConfig, TrlParser

from wd1.eval.countdown import CTDDataset
from wd1.eval.gsm8k import GSM8KDataset
from wd1.eval.math500 import MATH500Dataset
from wd1.eval.sudoku import SudokuDataset
from wd1.trainers.diffu_grpo_config import DiffuGRPOConfig
from wd1.trainers.diffu_grpo_trainer import DiffuGRPOTrainer
from wd1.trainers.eval_callback import AccuracyEvalCallback
from wd1.trainers.rev_grpo_ref_pol_trainer import RevDiffuRefPolGRPOTrainer
from wd1.trainers.rev_grpo_trainer import RevDiffuGRPOTrainer
from wd1.trainers.rev_grpo_trainer_psr import RevPSRDiffuGRPOTrainer

# Custom imports

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}

SUB_SAMPLE_MAP = {
    "gsm8k": 250,
    "math": 250,
    "countdown": -1,
    "sudoku": -1,
}


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main(grpo_config, model_config):
    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    # Load dataset based on configuration
    val_dataset = None
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        # small data for qucik test
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        # small data for qucik test
        reward_functions = [countdown_reward_func]
    elif grpo_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(
            range(0, len(dataset) - 500)
        )  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer - use SFT path if on top of SFT
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        grpo_config.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    val_dataset = DATASET_MAP[grpo_config.dataset](
        tokenizer,
        subsample=SUB_SAMPLE_MAP[grpo_config.dataset],
        num_examples=0,
        add_reasoning=True,  # prefill for all models
    )

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    if is_main_process():
        print("Trainer type is: ", grpo_config.trainer_type)

    # Initialize and run trainer
    if grpo_config.trainer_type == "wll_d1_neg":
        # NSR + PSR + d1 objective
        trainer = RevDiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "wll_d1_pos_only":
        trainer = RevPSRDiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "d1":
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    elif grpo_config.trainer_type == "wll_d1_neg_ref":
        # add reference policy regularisation for
        trainer = RevDiffuRefPolGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            eval_dataset=val_dataset,
            callbacks=[
                AccuracyEvalCallback(
                    val_dataset,
                    tokenizer=tokenizer,
                    gen_length=grpo_config.max_completion_length,
                    temperature=0.0,
                    steps=grpo_config.diffusion_steps,
                    block_length=grpo_config.block_length,
                    batch_size=grpo_config.per_device_eval_batch_size,
                )
            ],
        )
    else:
        raise Exception("Not know trainer type")

    if is_main_process():
        wandb.init(project=grpo_config.wandb_project, name=grpo_config.run_name)

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    main(grpo_config=grpo_config, model_config=model_config)
