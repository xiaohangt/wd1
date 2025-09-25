# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from open_r1.diff_grpo.trainer import DiffuGRPOTrainer
from open_r1.configs import DiffuGRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()
logger = logging.getLogger(__name__)

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )
    lora: bool = field(default=False, metadata={"help": "Use LoRa or not"})
    peft_task_type: str = field(default="CAUSAL_LM")
    num_train_samples: int = field(default=3000, metadata={"help": "Number of training samples"})


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name.rstrip("-debug"), name=script_args.dataset_config, cache_dir="cache")

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    tokenizer.mask_token_id = 126336 #fixed
    # tokenizer.pad_token_id = tokenizer.mask_token_id
    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            # prompt.append({"role": "system", "content": training_args.system_prompt})
            # LLaDA doesn't offer system prompt so we append the system prompt to the end of the user prompt
            example["problem"] += ("\n" + training_args.system_prompt)
        if "Base" in tokenizer.name_or_path:
            return {"prompt": example["problem"]}
        else:
            prompt.append({"role": "user", "content": example["problem"]})
            return {"prompt": prompt}

    dataset = dataset.map(make_conversation, keep_in_memory=True)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        cache_dir="./cache"
    )

    if script_args.dataset_name == "DigitalLearningGmbH/MATH-lighteval-debug":
        # The "special" data that we want to optimize for
        # import pandas as pd
        # df = pd.read_csv("MATH-lighteval.csv")
        # include_idx = df[(df["answer_correct"] == False) & (df["intermediate_correct"] == True)]["p_index"].unique().tolist()
        include_idx = [291, 47, 1334, 2126]
    elif script_args.dataset_name == "camel-ai/amc_aime_self_improving":
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].rename_column("groud_truth_solution", "solution")
        include_idx = list(range(len(dataset[script_args.dataset_train_split])))
    elif script_args.dataset_name in ["agentica-org/DeepScaleR-Preview-Dataset", "open-r1/OpenR1-Math-220k"]:
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].remove_columns(
            ["solution"])
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].rename_column(
            "answer", "solution")
        dataset[script_args.dataset_train_split] = dataset[script_args.dataset_train_split].remove_columns(
            [i for i in dataset[script_args.dataset_train_split].column_names if i not in ["problem", "solution", "prompt"]])
        include_idx = np.random.choice(np.arange(len(dataset[script_args.dataset_train_split])), size=script_args.num_train_samples,
                                       replace=False).tolist()
    else:
        include_idx = np.random.choice(np.arange(len(dataset[script_args.dataset_train_split])), size=script_args.num_train_samples,replace=False).tolist()
    if script_args.dataset_name == "DigitalLearningGmbH/MATH-lighteval-debug":
        eval_dataset = dataset[script_args.dataset_train_split].select((
        i for i in range(len(dataset[script_args.dataset_train_split]))
        if i in set(include_idx)
    ))
    else:
        eval_dataset = load_dataset("HuggingFaceH4/MATH-500", cache_dir="./cache")["test"]
        eval_dataset = eval_dataset.select(range(100, 150))
        eval_dataset = eval_dataset.map(make_conversation, keep_in_memory=True)
        eval_dataset = eval_dataset.remove_columns(["solution"])
        eval_dataset = eval_dataset.rename_column("answer", "solution")
    # training_args.model_init_kwargs = model_kwargs
    from llada.modeling_llada import LLaDAModelLM
    from llada.configuration_llada import LLaDAConfig
    model = LLaDAModelLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    if script_args.lora:
        peft_config = LoraConfig(
            task_type=script_args.peft_task_type,
            inference_mode=False,
            target_modules=["q_proj", "k_proj", "v_proj"],
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    # model_config = LLaDAConfig.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    # model_config.n_layers = 4
    # model = LLaDAModelLM(model_config)
    #############################
    # Initialize the DiffGRPO trainer
    #############################
    trainer = DiffuGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split].select((
        i for i in range(len(dataset[script_args.dataset_train_split]))
        if i in set(include_idx)
    )),
        eval_dataset=eval_dataset, #dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, DiffuGRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
