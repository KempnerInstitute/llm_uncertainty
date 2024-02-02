import copy
import functools
import json
import logging
import os
import pickle
import sys
import time
from typing import Optional
import warnings

import lightning as L
import torch
from transformers import (
    GPTNeoXForCausalLM,
    AutoTokenizer,
)

from lit_llama import LLaMA, Tokenizer
from lit_llama.model import pipeLLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice, jsd
from lit_gpt import GPT, Config, Tokenizer
from train_head_utils import (
    load_llama,
    load_llama_2,
    load_pythia,
    MAX_LEN,
)

#DTYPE = torch.float32
DTYPE = torch.bfloat16
DEVICE = torch.device('cuda:0')
SUPPORTED_MODEL_TYPES = set([
    "llama",
    "llama_2",
    "pythia",
])


def write_shard(outputs, shard_path, compress=False):
    extension = shard_path.split('.')[-1]
    if(not compress):
        assert(extension == "pickle")
        open_fn = open
    else:
        assert(extension == "xz")
        open_fn = lzma.open
   
    with open_fn(shard_path, "wb") as fp:
        pickle.dump(outputs, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote shard {shard_path}...")


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
    *,
    prompts_json_path: str,
    output_dir: str,
    checkpoint_path: str, 
    model_type: str = "llama",
    model_size: str = "30B",
    tokenizer_path: Optional[str] = None,
    quantize: Optional[str] = None,
    output_shard_size: int = 2500,
    return_embeddings: bool = False,
    return_initial_embeddings: bool = False,
    return_after_layer_n: Optional[int] = None,
    return_sum_all_layers: bool = False,
    resume: bool = False,
    revision: int = -1,
    compress: bool = False,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompts_json_path: A JSON file containing a dictionary of prompts keyed by prompt IDs
        output_dir: Where to save output pickle files
        model_type: Model class. e.g. "llama" or "pythia"
        model_size: The size of the model to use. E.g. "7B" or "30B"
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit model.
        output_shard_size: Number of outputs per output shard
        return_embeddings: Whether to skip the logit head and return raw embeddings
        return_initial_embeddings: Whether to immediately return the sequence embedding
        resume: Quick and dirty resume functionality. DON'T CHANGE HYPERPARAMS.
        revision: The version of Pythia to use
        compress: Whether to compress outputs
    """
    assert(model_type in SUPPORTED_MODEL_TYPES)

    if(model_type == "llama"):
        assert(tokenizer_path is not None)

    assert sum([return_embeddings, return_initial_embeddings, return_after_layer_n is not None, return_sum_all_layers]) <= 1, \
            "Only one return type may be enabled"

    # Create the output dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the model and tokenizer
    if(model_type == "llama"):
        model, tokenizer = load_llama(
            model_size, checkpoint_path, tokenizer_path, DTYPE, quantize, load_pipellama=False,
        )
    elif(model_type == "llama_2"):
        model, tokenizer, fabric = load_llama_2(
            model_size, checkpoint_path, DTYPE, return_fabric=True
        )
    elif(model_type == "pythia"):
        model, tokenizer = load_pythia(
            model_size, checkpoint_path, DTYPE, revision=revision,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    model.eval()

    # Load the prompts
    with open(prompts_json_path, "r") as fp:
        prompts = json.load(fp)

    prompts = list(sorted(prompts.items(), key=lambda t: t[0]))

    def get_shard_path(shard_count, compress=False):
        output_basename = os.path.split(prompts_json_path)[-1].split('.')[0]
        output_basename += f"_{model_type}"
        output_basename += f"_{model_size}"
        if(return_embeddings):
            output_basename += "_emb"
        shard_name = f"{output_basename}_{shard_count}.pickle"
        if(compress):
            shard_name += ".xz"
        return os.path.join(output_dir, shard_name)

    # Generate logits
    if(resume):
        print(f"Resuming computation...")

    shard_count = 0
    shard_path = get_shard_path(shard_count, compress=compress)
    skip = False
    outputs = {}
    for i, (key, prompt) in enumerate(prompts):
        # Write shard
        if(i != 0 and i % output_shard_size == 0):
            if(len(outputs)):
                write_shard(outputs, shard_path, compress=compress)

            shard_count += 1
            shard_path = get_shard_path(shard_count, compress=compress)
            outputs = {}

        # Skip precomputed shard entries
        if(resume and os.path.isfile(shard_path)):
            continue

        # Tokenize the prompt
        encoded_prompt = tokenizer(prompt)
        len_prompt = len(encoded_prompt)
        encoded_prompt = encoded_prompt.unsqueeze(0)  # add batch dimension

        if(len_prompt == 0):
            print(f'Skipping "{key}" (too short)...')
            continue

        if(model_type == "llama"):
            max_len = model.config.block_size
        elif(model_type == "llama_2"):
            max_len = model.config.block_size
        else:
            max_len = MAX_LEN

        if(len_prompt > max_len):
            logging.info(f'Truncating {key}...')
            encoded_prompt = encoded_prompt[..., :max_len]
            len_prompt = max_len

        # Run the model
        with torch.no_grad():
            if(model_type == "llama"):
                fn = model.forward
                if(return_embeddings):
                    fn = model._forward
                elif(return_initial_embeddings):
                    fn = model.embed_sequence
                elif(return_after_layer_n):
                    assert(isinstance(model, LLaMA)) # not the weird piped version
                    model.transformer.h = model.transformer.h[:return_after_layer_n + 1]
                    fn = model._forward
                elif(return_sum_all_layers):
                    def fn(idx):
                        x = model.embed_sequence(idx)
                        emb = 0
                        for h in model.transformer.h:
                            x = h(x)
                            emb += x

                        return emb
            elif(model_type == "llama_2"):
                #### THIS IS NECESSARY TO KEEP FSDP ENTROPIES CORRECT ####
                #### why is lightning the way that it is ####
                with fabric.init_tensor():
                    model.max_seq_length = encoded_prompt.shape[-1]
                ##########################################################

                fn = model.forward
                if(return_embeddings):
                    def fn(p):
                        _, embeddings = model.forward(
                            p, None, return_embeddings=True
                        )
                        return embeddings
                elif(return_initial_embeddings):
                    raise NotImplementedError
                elif(return_after_layer_n):
                    raise NotImplementedError
            elif(model_type == "pythia"):
                fn = pythia_forward(model)
                if(return_embeddings):
                    fn = pythia_forward(model, embeddings=True)
                elif(return_initial_embeddings):
                    fn = model.gpt_neox.embed_in
                elif(return_after_layer_n):
                    fn = pythia_forward(
                        model, 
                        embeddings=True, 
                        return_after_layer_n=return_after_layer_n
                    )

            logits = fn(encoded_prompt)

        logits = logits.squeeze(0)
        logits = logits.cpu()
        outputs[key] = logits

    if(len(outputs)):
        write_shard(outputs, shard_path, compress=compress)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    warnings.filterwarnings(
        # SLURM srun warning
        "ignore", 
        message="The `srun` command is available on your system but is not used",
    )

    CLI(main)
