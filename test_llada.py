import os.path

import torch
import itertools
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.llada.modeling_llada import LLaDAModelLM
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
import numpy as np
from src.llada.generate import get_num_transfer_tokens, add_gumbel_noise, get_num_transfer_tokens_maskgit
from src.open_r1.utils.trainer_utils import profiling_context, CustomDistributedSampler
import torch.distributed as dist
import torch.nn.functional as F
import random
import pandas as pd
from latex2sympy2_extended import NormalizationConfig
from tqdm import tqdm
# from visualize_diffusion import DiffusionModelVisualizer
from torch.utils.data import DataLoader

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()

@torch.no_grad()
def generate_dlm(model, prompt, steps=64, gen_length=128, block_length=32, temperature=0.,
                 cfg_scale=0., remasking='low_confidence', mask_id=126336, implicit_diffusion=False,
                 overtime_conf=False, mode="linear"):
    '''
    Optimized version of the generate function.
    '''
    # Use mixed precision for faster computation
    with torch.amp.autocast("cuda", enabled=True):
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        intermediate_inputs = []
        intermediate_results = []
        intermediate_confidence = []
        # Adjust steps if needed
        steps_per_block = max(1, steps // num_blocks)
        overtime_confidence = torch.zeros_like(x, dtype=torch.float32)
        if implicit_diffusion:
            logits = model(x, diffusion_steps=steps).logits
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x = torch.argmax(logits_with_noise, dim=-1)
        else:
            for num_block in range(num_blocks):
                start_idx = prompt.shape[1] + num_block * block_length
                end_idx = prompt.shape[1] + (num_block + 1) * block_length

                block_mask_index = (x[:, start_idx:end_idx] == mask_id)
                num_transfer_tokens = get_num_transfer_tokens_maskgit(block_mask_index, steps_per_block, mode=mode)

                for i in range(steps_per_block):
                    mask_index = (x == mask_id)
                    intermediate_inputs.append(x.clone().cpu())
                    # Handle classifier-free guidance more efficiently
                    if cfg_scale > 0.:
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
                    if remasking == 'low_confidence':
                        # Use float32 instead of float64 for better performance
                        p = F.softmax(logits, dim=-1)
                        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                    elif remasking == 'random':
                        x0_p = torch.rand(x0.shape, device=x0.device)
                    else:
                        raise NotImplementedError(remasking)
                    if not overtime_conf:
                        intermediate_confidence.append(x0_p.clone().cpu())
                    # Ensure we don't process tokens beyond the current block
                    x0_p[:, end_idx:] = -np.inf
                    # Update masked tokens
                    x0 = torch.where(mask_index, x0, x)
                    intermediate_results.append(x0.clone().cpu())
                    # valid_token_mask = x0 != 198
                    # confidence = torch.where(torch.logical_and(mask_index, valid_token_mask), x0_p, torch.tensor(-np.inf, device=x0.device))
                    confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                    # Select tokens to transfer based on confidence
                    for j in range(confidence.shape[0]):
                        num_tokens = num_transfer_tokens[j, i].item()
                        if overtime_conf:
                            # if confidence[j][mask_index[j]].min() > 0.3:
                            #     select_indices = (torch.where(confidence < 0.3, -np.inf, confidence)[j] != -np.inf).nonzero(as_tuple=True)[0]
                            # else:
                            #     len(confidence[j][mask_index[j]]) * 0.5
                            _, select_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, i:].sum().item())
                            # if len(select_indices) < num_tokens:
                            #     _, select_indices = torch.topk(confidence[j], k=num_tokens)
                            x[j, select_indices] = x0[j, select_indices]
                            overtime_confidence[j, select_indices] = confidence[j, select_indices].clone()
                            # if (x[j,:] == mask_id).sum() <= 0:
                            if i != (steps_per_block - 1):
                                overtime_conf_wo_zeros = \
                                    torch.where(overtime_confidence == 0.0, 1.0, overtime_confidence)[j]
                                num_tokens_to_mask = num_transfer_tokens[j, i + 1:].sum().item()
                                # if num_tokens_to_mask < 0:
                                #     break
                                # threshold_p = 0.9
                                # overtime_conf_wo_zeros = torch.where(overtime_conf_wo_zeros > threshold_p, 1.0,
                                #                                      overtime_conf_wo_zeros)
                                # if overtime_conf_wo_zeros[overtime_conf_wo_zeros < threshold_p].shape[
                                #     0] < num_tokens_to_mask:
                                #     num_tokens_to_mask = \
                                #     overtime_conf_wo_zeros[overtime_conf_wo_zeros < threshold_p].shape[0]
                                _, mask_select_indices = torch.topk(overtime_conf_wo_zeros, k=num_tokens_to_mask,
                                                                    largest=False)
                                if len(mask_select_indices) == 0:
                                    break
                                x[j, mask_select_indices] = mask_id
                        else:
                            if num_tokens > 0:
                                _, select_indices = torch.topk(confidence[j], k=num_tokens)
                                x[j, select_indices] = x0[j, select_indices]
                    if overtime_conf:
                        intermediate_confidence.append(overtime_confidence.clone().cpu())
        return x, intermediate_results, intermediate_confidence, intermediate_inputs

@torch.no_grad()
def generate_ar(model, prompt, steps=64, gen_length=128, **kwargs):
    generate_kwargs = {
        "max_new_tokens": gen_length,
        "min_new_tokens": 10,
        "temperature": 0.7,
        "do_sample": False,  # The three options below used together leads to contrastive search
        "top_k": 4,
        "penalty_alpha": 0.6,
        "use_cache": True
        # "no_repeat_ngram_size": no_repeat_ngram_size,
        # **generation_config,
    }
    generated_ids = model.generate(
        prompt,
        # stopping_criteria=stopping_criteria,
        **generate_kwargs
    )
    return generated_ids.cpu(), [], [], []

def visualize_intermediates(intermediates, intermediate_inputs, intermediate_correct_cnt, vis_file_name):
    # Create visualizer
    visualizer = DiffusionModelVisualizer(cmap_name='plasma')
    # Load data
    responses = []
    for response in intermediates:
        resp_tokens = tokenizer.convert_ids_to_tokens(response.cpu()[0, -args.gen_length:])
        new_resp_tokens = []
        for token in resp_tokens:
            if token == "Ċ":
                new_resp_tokens.append("Ċ")
            elif token == "Ġ":
                new_resp_tokens.append("Ġ")
            elif token.startswith("Ġ"):
                new_resp_tokens.append(token.lstrip("Ġ"))
            else:
                new_resp_tokens.append(token)
        responses.append(new_resp_tokens)
    inputs = []
    for input_tokens in intermediate_inputs:
        inp_tokens = tokenizer.convert_ids_to_tokens(input_tokens.cpu()[0, -args.gen_length:])
        new_inp_tokens = []
        for token in inp_tokens:
            if token == "Ċ":
                new_inp_tokens.append("Ċ")
            elif token == "Ġ":
                new_inp_tokens.append("Ġ")
            elif token.startswith("Ġ"):
                new_inp_tokens.append(token.lstrip("Ġ"))
            elif token == "<|mdm_mask|>":
                new_inp_tokens.append("[MASK]")
            else:
                new_inp_tokens.append(token)
        inputs.append(new_inp_tokens)
    confidence_scores = [
        torch.where(i[0, -args.gen_length:].cpu() == float("-inf"), 1, i[0, -args.gen_length:].cpu()).numpy().tolist()
        for i in confidences]
    visualizer.load_data(responses, confidence_scores,
                         ["Correct" if i in intermediate_correct_cnt else "Wrong" for i in range(len(inputs))], inputs=inputs)
    # Create web visualization
    visualizer.create_web_visualization(vis_file_name)

def parse_solution(solution):
    gold_parsed = parse(
        solution,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )
    if len(gold_parsed) == 0:
        gold_parsed = parse(
            "$" + solution + "$",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    return gold_parsed

if __name__ == '__main__':
    local_rank = setup_ddp()
    device = local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HuggingFaceH4/MATH-500", choices=["openai/gsm8k", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/aime_2024", "HuggingFaceH4/MATH-500"])
    parser.add_argument("--method_name", default="wd1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--system_prompt_type", default="normal")
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--lora_path", default=None, type=str)
    parser.add_argument("--model_type", default="dlm")
    parser.add_argument("--mode", default="linear", choices=["linear", "cosine", "pow2", "pow3", "pow0.5", "log", "exp"])
    parser.add_argument("--log_visualizations", default=False, action="store_true")
    parser.add_argument("--overtime_conf", default=False, action="store_true")
    args = parser.parse_args()

    # model_path = "data/LLaDA-8B-Instruct-GDPO-random-test-diff-reward/"
    # model_path = "data/LLaDA-8B-Instruct-GDPO-numina-adv-v3"
    # model_path = "GSAI-ML/LLaDA-8B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, cache_dir="./cache")
    except:
        tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, cache_dir="./cache")
    if args.model_type == "dlm":
        model = LLaDAModelLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                          cache_dir="./cache", device_map=device)
        if args.lora_path is not None:
            model.load_adapter(args.lora_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, cache_dir="./cache", torch_dtype=torch.bfloat16, device_map=device)
    if args.system_prompt_type == "format":
        system_prompt = "Let's first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer> and output the final answer within \\boxed{} inbetween the <answer> </answer> tags"
    elif args.system_prompt_type == "step_by_step":
        system_prompt = "Let's think step by step and output the final answer within \\boxed{}."
    elif args.system_prompt_type == "d1":
        system_prompt = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. Respond in the following format: <reasoning> Your reasoning here </reasoning> <answer> \\boxed{...} </answer>" """
    else:
        system_prompt = "Solve this problem and output the final answer within \\boxed{}."
    # dataset_name = "agentica-org/DeepScaleR-Preview-Dataset" #HuggingFaceH4/MATH-500, HuggingFaceH4/aime_2024, agentica-org/DeepScaleR-Preview-Dataset
    # ds = load_dataset("open-r1/OpenR1-Math-220k", cache_dir="./cache")["train"]
    # ds = load_dataset("HuggingFaceH4/aime_2024", cache_dir="./cache")["train"]
    # ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", cache_dir="./cache")["train"]
    dataset_name = args.dataset_name
    method_name = args.method_name
    if dataset_name == "DigitalLearningGmbH/MATH-lighteval":
        ds = load_dataset(dataset_name, cache_dir="./cache")["test"]
        import pandas as pd

        df = pd.read_csv("MATH-lighteval.csv")
        include_idx = df[(df["answer_correct"] == False) & (df["intermediate_correct"] == True)][
            "p_index"].unique().tolist()
        include_idx = pd.read_csv("MATH-lighteval_Llada_original.csv")["p_index"].unique().tolist()
        ds = ds.select((
            i for i in range(len(ds))
            if i in set(include_idx)
        ))
    elif dataset_name == "agentica-org/DeepScaleR-Preview-Dataset":
        ds = load_dataset(dataset_name, cache_dir="./cache")["train"]
        ds = ds.remove_columns(["solution"])
        ds = ds.rename_column("answer", "solution")
    elif dataset_name == "openai/gsm8k":
        ds = load_dataset(dataset_name, "main", cache_dir="./cache")[args.split]
        ds = ds.rename_column("question", "problem")
        ds = ds.rename_column("answer", "solution")
    else:
        ds = load_dataset(dataset_name, cache_dir="./cache")[args.split]
        # include_idx = [0,1,2,3,4,5,6,7,8,9] #[6] #93, 46, 19 some hard sample that we can use to test our idea
        # ds = ds.select((
        #     i for i in range(len(ds))
        #     if i in set(include_idx)
        # ))
    all_results = []
    dataloader = DataLoader(
        ds,
        batch_size=1,
        sampler=CustomDistributedSampler(ds, shuffle=False),
    )
    for p_index, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # problem_index = random.randint(0, len(ds) - 1)
        # problem_index = 5371
        # problem, answer, solution = ds[problem_index]["problem"], ds[problem_index]["answer"], ds[problem_index]['solution']

        problem, solution = d["problem"][0], d["solution"][0]
        unique_id = d.get("unique_id", [p_index])[0]
        unique_id = unique_id.replace("/", "_").rstrip(".json") if isinstance(unique_id, str) else unique_id
        level = d.get('level', [1])[0]
        p_type = d.get('type', ['math'])[0]
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            gold_parsed = parse(
                "$" + solution + "$",
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
        problem += "\n"
        problem += system_prompt
        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        m = [{"role": "user", "content": problem}, ]
        prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        if args.model_type == "dlm":
            generate = generate_dlm
            # sampling_settings = [(1, 512), (4, 512), (16, 512), (32, 512), (128, 512), (512, 512),
            #                      (4, 256), (16, 256), (32, 256), (128, 256), (512, 256),
            #                      (4, 128), (16, 128), (32, 128), (128, 128), (512, 128),
            #                      (16, 64), (32, 64), (128, 64), (512, 64)
            #                      ] # A list of (block_length, step)
            # sampling_settings = [(128, 64), (128, 128), (128, 256), (32, 64), (32, 128), (32, 256)]
            block_sizes = [128]
            steps = [256]
        else:
            generate = generate_ar
            block_sizes = [args.gen_length]
            steps = [1]
        for block_size in block_sizes:
            for step in steps:
        # for block_size, step in sampling_settings:
                if step % (args.gen_length / block_size) != 0:
                    break
                out, intermediates, confidences, intermediate_inputs = generate(model, input_ids, steps=step, gen_length=args.gen_length, block_length=block_size, temperature=0., cfg_scale=0.,
                               remasking='low_confidence', implicit_diffusion=False, overtime_conf=args.overtime_conf, mode=args.mode)
                model_answer = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                if len(gold_parsed) != 0:
                    # We require the answer to be provided in correct latex (no malformed operators)
                    answer_parsed = parse(
                        model_answer,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    equations=True,
                                    boxed="all",
                                    units=True,
                                ),
                                # Ensures that boxed is tried first
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                    intermediate_answers = tokenizer.batch_decode(
                        torch.cat(intermediates, dim=0)[:, input_ids.shape[1]:],
                        skip_special_tokens=True)
                    answer_correct = verify(answer_parsed, gold_parsed)
                    # print(f"Question {problem_index} is {str(answer_correct)}")
                    # intermediate_correct = False
                    intermediate_correct_cnt = []
                    for i, intermediate_answer in enumerate(intermediate_answers):
                        intermediate_parsed = parse(
                            intermediate_answer,
                            extraction_config=[
                                LatexExtractionConfig(
                                    normalization_config=NormalizationConfig(
                                        nits=False,
                                        malformed_operators=False,
                                        basic_latex=True,
                                        equations=True,
                                        boxed="all",
                                        units=True,
                                    ),
                                    # Ensures that boxed is tried first
                                    boxed_match_priority=0,
                                    try_extract_without_anchor=False,
                                )
                            ],
                            extraction_mode="first_match",
                        )
                        if verify(gold_parsed, intermediate_parsed):
                            # intermediate_correct = True
                            intermediate_correct_cnt.append(i)
                        # if verify(gold_parsed, intermediate_parsed) and not answer_correct:
                        #     print(f"Correct prediction at timestep {i} for question {problem_index}")
                    if (not answer_correct) and len(intermediate_correct_cnt) > 0 and args.log_visualizations:
                        vis_file_name = f"logs/visualizations/htmls/{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_prompt_{args.system_prompt_type}_{args.mode}_{step}_{block_size}_{unique_id}_RCR_{str(args.overtime_conf)}.html"
                        # visualize_intermediates(intermediates, intermediate_inputs, intermediate_correct_cnt, vis_file_name)
                    all_results.append({"id": unique_id,"problem": problem, "solution": solution, "model_answer": model_answer, "level": level,
                                        "p_type": p_type, "block_size": block_size, "step": step,
                                        "answer_correct": answer_correct, "intermediate_correct": intermediate_correct_cnt})
    dist.barrier()
    file_name = f"./local_rank_{dist.get_rank()}_{method_name}_{dataset_name.split('/')[-1]}_{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_prompt_{args.system_prompt_type}_{args.mode}_{args.gen_length}_RCR_{str(args.overtime_conf)}.csv"
    pd.DataFrame(all_results).to_csv(os.path.join("./logs", file_name), index=False)
    if dist.get_rank() == 0:
        dfs = []
        all_file_name = file_name = f"./{method_name}_{dataset_name.split('/')[-1]}_{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_prompt_{args.system_prompt_type}_{args.mode}_{args.gen_length}_RCR_{str(args.overtime_conf)}.csv"
        for rank in range(dist.get_world_size()):
            file_name = f"./local_rank_{rank}_{method_name}_{dataset_name.split('/')[-1]}_{args.model_path.rstrip('/').split('/')[-1] if args.lora_path is None else args.lora_path.rstrip('/').split('/')[-1]}_prompt_{args.system_prompt_type}_{args.mode}_{args.gen_length}_RCR_{str(args.overtime_conf)}.csv"
            dfs.append(pd.read_csv(os.path.join("./logs", file_name)))
            os.remove(os.path.join("./logs", file_name))
        pd.concat(dfs).to_csv(os.path.join("./logs", all_file_name), index=False)
    cleanup_ddp()