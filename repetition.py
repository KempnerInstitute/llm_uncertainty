import sys, os
import time
from typing import Optional

from jsonargparse import CLI
import torch
import torch.nn as nn

import json
import pickle
from pathlib import Path
from tqdm import tqdm
from train_head_utils import (
    load_llama,
    load_pythia,
    MAX_LEN,
    load_lm_head,
)
from transformers import (
    AutoTokenizer,
)
from lit_llama import Tokenizer
import matplotlib.pyplot as plt
import numpy as np


DEVICE= "cuda"
# DTYPE = torch.bfloat16 if torchs.cuda.is_bf16_supported() else torch.float32
DTYPE = torch.float32

def log(verbose, *content, **kwargs):
    if verbose:
        print(*content, **kwargs)

def pythia_forward(model, embeddings=False, return_after_layer_n=-1):
    def fw(input_ids):
        outputs = model.gpt_neox(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[return_after_layer_n]

        if(embeddings):
            return hidden_states

        logits = model.embed_out(hidden_states)
        return logits

    return fw

def main(
    repetition_filter: str,
    shard_count: str, 
    prompts_json_path: str,
    model_type: str,
    model_size: str,
    checkpoint_path: str,
    experiment_name: Optional[str] = None,
    k: int = 10,
    tokenizer_path: Optional[str] = None,
    sample_until_period: bool = True,
    addl_token_limit: int = 100,
    example_path: Optional[str] = None, 
):
    if(model_type == "llama"):
        assert(tokenizer_path is not None)

    assert checkpoint_path.exists()
    if model_type == "llama": assert tokenizer_path.is_file()

    # Initialize the model and tokenizer
    if(model_type == "llama"):
        model, tokenizer = load_llama(
            model_size, checkpoint_path, tokenizer_path, DTYPE, None
        )
    elif(model_type == "pythia"):
        model, tokenizer = load_pythia(
            model_size, checkpoint_path, DTYPE
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    model.eval()

    # load prompt indicator 
    shard_count = int(shard_count)
    prompt_filter_fp = os.path.join(repetition_filter, "filter", f"filter_{shard_count}.pickle")
    # prompt_loader = PrecomputedShardLoader([prompt_filter_fp])
    with open(prompt_filter_fp, "rb") as fp:
        prompt_loader = pickle.load(fp)
            
    large_entropy_dict_path = f"{repetition_filter}/large_entropy/large_entropy_{shard_count}.pickle"
    with open(large_entropy_dict_path, "rb") as fp:
        large_entropy_dict = pickle.load(fp)
    small_entropy_dict_path = f"{repetition_filter}/small_entropy/small_entropy_{shard_count}.pickle"
    with open(small_entropy_dict_path, "rb") as fp:
        small_entropy_dict = pickle.load(fp)

    with open(prompts_json_path, "r") as fp:
        prompts = json.load(fp)
    
    encoded_prompts = []
    prompt_type = []
    promot_indicator_sum = 0
    large_entropy_all = []
    small_entropy_all = []
    for i, (prompt_key, promot_indicator) in enumerate(prompt_loader.items()):
        if promot_indicator.sum()  == 0: continue
        # print(prompt_key, promot_indicator.sum())
        if model_type == "llama":
            encoded_prompt = tokenizer(prompts[prompt_key])
        elif model_type == "pythia":
            encoded_prompt = tokenizer(prompts[prompt_key])
        
        # encoded_prompt = encoded_prompt[:MAX_LEN]
        large_entropy_array = large_entropy_dict[prompt_key]
        small_entropy_array = small_entropy_dict[prompt_key]
        try:
            assert promot_indicator.shape[-1] == encoded_prompt.shape[0]
            assert promot_indicator.shape[-1] == large_entropy_array.shape[0] == small_entropy_array.shape[0]
        except:
            print("Exception!! Shape Mismatch - ignoring... ", promot_indicator.shape, encoded_prompt.shape)
            continue
        promot_indicator_sum += promot_indicator.sum()

        for eligible_index in torch.argwhere(promot_indicator):
            if eligible_index < MAX_LEN:
                assert promot_indicator[eligible_index] == 1
                encoded_prompts.append(encoded_prompt[None, :eligible_index+1]) # confirmed index is correct
                
                large_entropy = large_entropy_array[eligible_index].double()
                large_entropy_all.append(large_entropy)
                
                small_entropy = small_entropy_array[eligible_index].double()
                small_entropy_all.append(small_entropy)
    
                prompt_type.append(large_entropy>0.2)
                # small_entropy = small_entropy_array[eligible_index].double()
    prompt_type = torch.LongTensor(prompt_type)

    print(f"{len(prompt_type)} encoded prompts , w/ {prompt_type.sum()} low_e_high_a examples.", file=sys.stderr) 

    if experiment_name is not None:
        save_dir = os.path.join(repetition_filter, experiment_name)
        os.makedirs(save_dir, exist_ok=True)

    else:
        save_dir = repetition_filter

    if model_type == "pythia":
        tokenizer= AutoTokenizer.from_pretrained(
            f"EleutherAI/pythia-{model_size}",
        )
    elif model_type == "llama":
        tokenizer = Tokenizer(tokenizer_path)

    small_lm_head = load_lm_head(
        checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=model_size
    )
    
    # repetition experiment
    new_embed_all = []
    orig_embed_all = []
    encoded_prompt_all = []
    t0 = time.time()
    for i, encoded_prompt in enumerate(encoded_prompts):
        if i != 0 and i % 100 == 0: 
            print(f"{i}, {time.time() - t0:.02f} seconds.", file=sys.stderr)
            t0 = time.time()
        sys.stdout.flush()
        torch.cuda.empty_cache() 
        orig_embed, repetition_embeds = repetition_experiment(model, model_type, small_lm_head, encoded_prompt, tokenizer, k,
                                                              sample_until_period=sample_until_period,
                                                              addl_token_limit=addl_token_limit,
                                                              example_path=example_path)
        if ~prompt_type.bool()[i]:
            log(example_path, "high_e_low_a example: ")
        else:
            log(example_path, "low_e_high_a example: ")
        orig_embed_all.append(orig_embed)
        new_embed_all.append(repetition_embeds)
        encoded_prompt_all.append(encoded_prompt.squeeze())

    new_embed_all = torch.concatenate(new_embed_all)
    orig_embed = torch.concatenate(orig_embed_all)
    torch.save({"new_embed": new_embed_all, 
                "original_embed": orig_embed_all, 
                "large_entropy": large_entropy_all, 
                "prompt_type": prompt_type,
                "encoded_prompt": encoded_prompt_all}, 
                f'{save_dir}/repetition_{shard_count}.pt')
    
    return


def repetition_experiment(model, model_type, small_lm_head, encoded_prompt, tokenizer, k, 
                          sample_until_period=True,
                          addl_token_limit=100,
                          verbose=False, 
                          ):
    
    len_prompt = encoded_prompt.shape[-1]
    log(verbose, f"\nPrompt: \n {tokenizer.decode(encoded_prompt[0])}", end=" ")
    # Run the model
    if model_type == "llama":
        with torch.no_grad():
            orig_embed = model._forward(encoded_prompt).detach()
            generated = model.lm_head(orig_embed).detach()
    elif model_type == "pythia":
        with torch.no_grad():
            orig_embed = pythia_forward(model, embeddings=True)(encoded_prompt)
            generated = small_lm_head(orig_embed)

    entropy = compute_entropy(generated[0, -1, :])
    log(verbose, f"(orinal entropy: {entropy:.3f})")   
    generated = torch.softmax(generated, dim=-1).detach().cpu()

    orig_embed = orig_embed[0, -1, :].detach().cpu()        

    # Top k tokens
    log(verbose, "\nTop K Token: \n")
    top_k = torch.topk(generated, k, dim=-1).indices[0, -1, :].to(DEVICE)

    for t in torch.unbind(top_k):
        log(verbose, f"{tokenizer.decode(t)}: {generated[0, -1, t]:.3f}", end=" ")

    log(verbose, "\n \nTop K Repetition: \n")

    # A surprise tool that will help us later
    # (a lone period produces a different token ID for some idiotic reason)
    if model_type == "llama":
        period_id = tokenizer.encode("Period.", bos=False, eos=False, device=DEVICE)[-1].item()
        eos_id = tokenizer.eos_id
    elif model_type == "pythia":
        period_id = tokenizer.encode("Period.")[-1]
        eos_id = tokenizer.encode("<|endoftext|>")[0]
    
    repetition_embeds = []
    for t in torch.unbind(top_k):
        prompt_with_candidate = torch.cat(
            [
                encoded_prompt,
                t[None, None],
            ],
            dim=-1
        )

        if(sample_until_period):
            addl_tokens = 0
            while True:
                if model_type == "llama":
                    with torch.no_grad():
                        repetition_logits = model.forward(prompt_with_candidate).detach().cpu()
                elif model_type == "pythia":
                    with torch.no_grad():
                        repetition_logits = pythia_forward(model)(prompt_with_candidate).detach().cpu()

                # repetition_logits = model.forward(prompt_with_candidate).detach().cpu()
                best_token = torch.argmax(repetition_logits, dim=-1)[:, -1].to(DEVICE)
                prompt_with_candidate = torch.cat(
                    [
                        prompt_with_candidate,
                        best_token[:, None],
                    ],
                    dim=-1
                )

                if(best_token == period_id or best_token == eos_id):
                    break

                addl_tokens += 1
                if(addl_tokens >= addl_token_limit):
                    break

        log(verbose, "[prompt]", tokenizer.decode(prompt_with_candidate[0, len_prompt:]), end="<EOS> [prompt]")

        if model_type == 'pythia':
            repetition_prompt = torch.cat(
                    [
                        torch.tensor(eos_id, device=DEVICE)[None, None],
                        prompt_with_candidate,
                        torch.tensor(period_id, device=DEVICE)[None, None],
                        torch.tensor(eos_id, device=DEVICE)[None, None],
                        encoded_prompt,
                    ],
                    dim=-1
                )
        elif model_type == 'llama':
            repetition_prompt = torch.cat(
                    [
                        prompt_with_candidate,
                        torch.tensor(tokenizer.eos_id, device=DEVICE)[None, None],
                        encoded_prompt,
                    ],
                    dim=-1
                )

        if model_type == 'llama':
            with torch.no_grad():
                repetition_embed = model._forward(repetition_prompt)[:, -1, :].detach().cpu()
                repetition_embeds.append(repetition_embed)
                if True:
                    repetition_logits = model.lm_head(repetition_embed.to(DEVICE))[0, :].detach()
                    entropy = compute_entropy(repetition_logits)
                    repetition_logits = torch.softmax(repetition_logits, dim=-1)
                    repetition_top_k = torch.topk(repetition_logits, k, dim=-1).indices
                    decoded = [tokenizer.decode(rt) for rt in torch.unbind(repetition_top_k)]
                    prob = [float(repetition_logits[rt]) for rt in repetition_top_k]
                    for d, p in zip(decoded, prob):
                        log(verbose, f"{d}: {p:.3f}", end=" ")
                    log(verbose, f"(new entropy: {entropy:.3f})") 
                    log(verbose, "\n")    
        elif model_type == "pythia":
            with torch.no_grad():
                repetition_embed = pythia_forward(model, embeddings=True)(repetition_prompt)[:, -1, :].detach().cpu()
                repetition_embeds.append(repetition_embed)
                if True:
                    repetition_logits = model.embed_out(repetition_embed.to(DEVICE))[0, :].detach()
                    repetition_logits = torch.softmax(repetition_logits, dim=-1)
                    repetition_top_k = torch.topk(repetition_logits, k, dim=-1).indices
                    decoded = [tokenizer.decode(rt) for rt in torch.unbind(repetition_top_k)]
                    prob = [float(repetition_logits[rt]) for rt in repetition_top_k]
                    for d, p in zip(decoded, prob):
                        log(verbose, f"{d}: {p:.5f}", end=" ")
                    log(verbose, "\n")    

    return orig_embed, torch.concatenate(repetition_embeds)[None, :]


def compute_entropy(logits):
    logits_softmax = torch.nn.functional.softmax(logits, dim=-1)
    logs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = torch.sum(-1 * logits_softmax * logs, dim=-1)
    return entropy



if __name__ == "__main__":

    from jsonargparse import CLI

    CLI(main)
