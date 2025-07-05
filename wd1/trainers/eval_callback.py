import re

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate.utils import (
    gather_object,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    disable_bar=True,
):
    """
    Optimized version of the generate function.
    """
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=prompt.device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=disable_bar, leave=False):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, torch.tensor(-np.inf, device=x0.device)
                )

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
        return x


class AccuracyEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_dataset,
        tokenizer,
        gen_length=128,
        temperature=0.0,
        steps=64,
        block_length=32,
        batch_size=4,
    ):
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.temperature = temperature
        self.steps = steps
        self.block_length = block_length

        self.eval_dataset = eval_dataset
        self.dataloader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            collate_fn=eval_dataset.collate_fn,
            drop_last=True,
        )

    def on_evaluate(self, args, state, control, **kwargs):
        accelerator = kwargs["accelerator"]
        model = kwargs["model"]
        # Split dataset across GPUs
        eval_dataloader = accelerator.prepare(self.dataloader)

        # Generate single completion for each prompt
        all_generations = []
        if accelerator.is_main_process:
            eval_dataloader = tqdm(eval_dataloader, desc="Evaluating", leave=True)

        for batch in eval_dataloader:
            input_ids = batch["input_ids"]
            gt_answers = batch["answers"]
            questions = batch["questions"]
            prompts = batch["prompts"]

            with torch.no_grad():
                out = generate(
                    model,
                    input_ids,
                    self.tokenizer,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    disable_bar=accelerator.is_main_process,
                )

            generated_texts = self.tokenizer.batch_decode(
                out[:, -self.gen_length :], skip_special_tokens=False
            )
            example_result = [
                {
                    "question": questions[j],
                    "prompt_input": prompts[j],
                    "generations": generated_texts[j],
                    "ground_truth": gt_answers[j],
                }
                for j in range(len(gt_answers))
            ]
            all_generations.extend(example_result)

        # Compute accuracy
        parsed_answer = None
        all_generations = gather_object(all_generations)
        if accelerator.is_main_process:
            total_correct = 0
            for example_result in all_generations:
                raw_generation = example_result["generations"]
                ground_truth = example_result["ground_truth"]

                boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
                if boxed_matches:
                    for boxed_content in boxed_matches:
                        boxed_content = boxed_content.strip()
                        if (
                            boxed_content
                            and boxed_content != "..."
                            and not re.match(r"^\.+$", boxed_content)
                        ):
                            try:
                                parsed_answer = float(boxed_content)
                                break
                            except ValueError:
                                numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                                if numbers:
                                    try:
                                        parsed_answer = float(numbers[0])
                                        break
                                    except ValueError:
                                        pass

                if parsed_answer is None:
                    answer_match = re.search(
                        r"<answer>(.*?)</answer>", raw_generation, re.DOTALL
                    )
                    if answer_match:
                        answer_text = answer_match.group(1).strip()
                        if answer_text:
                            try:
                                parsed_answer = float(answer_text)
                            except ValueError:
                                numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                                if numbers:
                                    try:
                                        parsed_answer = float(numbers[-1])
                                    except ValueError:
                                        pass

                is_correct = parsed_answer is not None and parsed_answer == ground_truth
                if is_correct:
                    total_correct += 1
            accuracy = total_correct / len(all_generations)
            # Log to wandb if enabled
            if args.report_to and "wandb" in args.report_to:
                wandb.log({"eval/accuracy": accuracy})
                print("Accuracy: ", accuracy)
                metrics = {"accuracy": accuracy}
                accelerator.log(metrics, step=state.global_step)

        # Synchronize all processes
        accelerator.wait_for_everyone()
