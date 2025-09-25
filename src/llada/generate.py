import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    '''
    if temperature == 0.:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def gamma_func(r, mode="cosine", total_num=512):
    # from https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py#L90 and https://github.com/google-research/maskgit/blob/main/maskgit/libml/mask_schedule.py#L21
    if mode == "linear":
        mask_ratio = 1 - r
    elif mode == "cosine":
        mask_ratio = np.cos(r * np.pi / 2)
    elif "pow" in mode:
        exponent = float(mode.replace("pow", ""))
        mask_ratio = 1 - r ** exponent
    elif mode == "log":
        mask_ratio = -np.log2(r) / np.log2(total_num)
    elif mode == "exp":
        mask_ratio = 1 - np.exp2(-np.log2(total_num) * (1-r))
    else:
        raise NotImplementedError
    mask_ratio = np.clip(mask_ratio, 1e-6, 1)
    return mask_ratio

def get_num_transfer_tokens_maskgit(mask_index, steps, mode="linear"):
    '''
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    '''
    total_num = mask_index.sum(dim=1, keepdim=True)
    #TODO: support batch_size>1 for gamma_func
    ratios = [[gamma_func((t+1) / steps, mode=mode, total_num=total_num_item.item()) for t in range(steps)] for total_num_item in total_num[:, 0]]
    num_transfer_tokens = total_num.expand((total_num.shape[0], steps))
    mask_ratios = torch.tensor(ratios).to(mask_index.device)
    num_transfer_tokens = total_num - torch.floor(num_transfer_tokens * mask_ratios)
    num_transfer_tokens = torch.cat([num_transfer_tokens[:, 0:1], num_transfer_tokens[:, 1:] - num_transfer_tokens[:, :-1]], axis=1)
    return num_transfer_tokens.to(torch.int64)

def get_num_transfer_tokens(mask_index, steps):
    '''
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    '''
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
def generate(model, prompt, steps=64, gen_length=128, block_length=32, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Optimized version of the generate function.
    '''
    # Use mixed precision for faster computation
    with torch.cuda.amp.autocast(enabled=True):
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        # Adjust steps if needed
        steps_per_block = max(1, steps // num_blocks)

        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = (x[:, start_idx:end_idx] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = (x == mask_id)

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

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]

        return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16, cache_dir="./cache").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, cache_dir="./cache")

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()