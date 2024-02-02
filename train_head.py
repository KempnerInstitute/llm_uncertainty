import copy
import json
import logging
import os
from pathlib import Path
import pickle
import random
import sys
import time
import warnings

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import wandb

from lit_llama import LLaMA, Tokenizer
from train_head_utils import (
    batch_loader,
    DistancePredictionHead,
    DistancePredictionHeadWithLMHead,
    entropy_threshold_acc,
    le_loss_term_loss,
    load_embedding_layer,
    load_lm_head,
    PrecomputedShardLoader,
    underestimate_loss_term_loss,
    _preprocessor,
)


DTYPE = torch.float32
DEVICE = torch.device("cuda:0")
SUPPORTED_MODEL_TYPES = set([
    "llama",
    "llama_2",
    "pythia",
])


# WandB x axis
WANDB_STEP_METRICS = set(["step", "epoch"])
# WandB y, x pairs
wandb_metrics = set([
    ("train_loss", "step"),
    ("train_accuracy", "step"),
    ("val_loss", "step"),
    ("val_unweighted_loss", "step"),
    ("val_accuracy", "step"),
    ("val_set_size", "step"),
    ("val_entropy_threshold_acc", "step"),
    ("val_small_entropy_loss", "step"),
])

ENTROPY_THRESHOLDS_FOR_METRICS = [1, 2, 3]
for le in ENTROPY_THRESHOLDS_FOR_METRICS:
    wandb_metrics.add((f"val_mse_below_{le}", "step"))

def _wandb_setup(args):
    wandb_args = {
        "project": "wandb_project",
        "entity": "wandb_entity",
        "name": "wandb_run_name",
        "dir": "output_dir",
    }
    for arg in wandb_args.values():
        if(args[arg] is None):
            raise ValueError(f"Must provide {arg} if use_wandb is True")

    wandb.login()

    wandb_config = {k:v for k,v in args.items()}
    slurm_jobid = os.environ["SLURM_JOBID"]
    if(slurm_jobid):
        wandb_config["slurm_jobid"] = slurm_jobid

    slurm_nodename = os.environ["SLURMD_NODENAME"]
    if(slurm_nodename):
        wandb_config["slurm_nodename"] = slurm_nodename

    wandb_run = wandb.init(
        config=wandb_config,
        **{k:args[v] for k, v in wandb_args.items()},
    )

    for step_metric in WANDB_STEP_METRICS:
        wandb.define_metric(step_metric)

    for metric, step_metric in wandb_metrics:
        assert(step_metric in WANDB_STEP_METRICS)
        wandb.define_metric(metric, step_metric=step_metric)

    # Save the git diff for reproducibility
    git_diff_path = os.path.join(args[wandb_args["dir"]], "git_diff.txt")
    os.system(f"git diff > {git_diff_path}")
    wandb.save(git_diff_path, base_path=f"./{args[wandb_args['dir']]}")

    return wandb_run


def _wandb_log(metrics, step_metric, step):
    assert step_metric in WANDB_STEP_METRICS
    for metric in metrics:
        assert (metric, step_metric) in wandb_metrics, \
            f"Metric {metric} not defined in the `metrics' dict"

    metrics = {
        **metrics,
        step_metric: step,
    }

    wandb.log(metrics)


def _validate_args(args):
    if(args["glue_lm_head"] and args["provide_entropy_as_input"]):
        raise ValueError("Cannot provide entropy as input using glue_lm_head")
    
    if(args["le_loss_term"] and (args["target_fn_name"] != "large_entropy" or args["bin_target"])):
        raise ValueError("le_loss_term must be used with target_fn_name=large_entropy and bin_target=False")
    
    if(args["underestimate_loss_term"] and (args["target_fn_name"] != "large_entropy" or args["bin_target"])):
        raise ValueError("underestimate_loss_term must be used with target_fn_name=large_entropy and bin_target=False")
    
    if(args["underestimate_loss_term"] and args["le_loss_term"]):
        raise ValueError("Cannot use underestimate_loss_term and le_loss_term simultaneously")


def main(
    *,
    precomputed_small_emb_dir: str,
    precomputed_large_emb_dir: str,
    output_dir: str,
    small_checkpoint_path: str,
    large_checkpoint_path: str,
    precomputed_head_input_emb_dir: str = None,
    small_model_size: str = "7B",
    large_model_size: str = "30B",
    model_type: str = "llama",
    hidden_dim: int = 2048,
    no_hidden_layers: int = 5,
    dropout: float = 0.1,
    activation: str = "relu",
    lr: float = 1e-6,
    batch_size: int = 64,
    no_epochs: int = 10,
    skip_frac: float = 0.95,
    nonzero_bin_weight: float = 1.,
    bin_target: bool = True,
    no_bins: int = 2,
    min_bin: float = -7,
    max_bin: float = np.log(np.log(2)), # JSD is bounded by ln(2)
    target_fn_name: str = "log_jsd",
    glue_lm_head: bool = False,
    use_logits_as_input: bool = False,
    softmax_input_logits: bool = False,
    seed: int = 42,
    min_entropy: float = None,
    max_entropy: float = None,
    provide_entropy_as_input: bool = False,
    append_predicted_token_embedding: bool = False,
    le_loss_term: bool = False,
    underestimate_loss_term: bool = False,
    upsample_low_large_entropy: bool = False,
    precomputed_small_emb_dir_val: str = None,
    precomputed_large_emb_dir_val: str = None,
    precomputed_head_input_emb_dir_val: str = None,
    dataset_filter_path: str = None,
    val_dataset_filter_path: str = None,
    small_model_revision: int = -1,
    eval_every_n_batches: int = 1000,
    resume_from: str = None,
    use_wandb: bool = False,
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_run_name: str = None,
) -> None:
    """
    Args:
        precomputed_small_emb_dir: Directory containing embeddings for the small model (generated by precompute_logits.py)
        precomputed_large_emb_dir: Directory containing embeddings for the large model (generated by precompute_logits.py)
        output_dir: Where to save output files
        precomputed_head_input_emb_dir: Directory containing embeddings to pass as input to the head, if different from precomputed_small_emb_dir (generated by precompute_logits.py)
        small_checkpoint_path: The small LM checkpoint path.
        large_checkpoint_path: The large LM checkpoint path.
        hidden_dim: Hidden dimension of the distance prediction head.
        no_hidden_layers: Number of hidden layers in the distance prediction head.
        dropout: Dropout probability in the distance prediction head.
        activation: Activation function in the distance prediction head.
        lr: Learning rate.
        batch_size: Batch size.
        no_epochs: Number of epochs.
        skip_frac: Probability of skipping any given token.
        nonzero_bin_weight: Nonzero bins are nonzero_bin_weight times more likely to be included in each batch.
        no_bins: Number of bins to discretize the target into.
        min_bin: Minimum value of the discretized target.
        max_bin: Maximum value of the discretized target.
        target_fn_name: Quantity to predict. One of "log_jsd" or "small_entropy".
        glue_lm_head: Whether to attach the small LLM's language modeling head to the head being trained here.
        seed: Random seed.
        le_loss_term: Whether to multiply the (bounded) inverse of the large model's entropy to MSE loss.
        precomputed_small_emb_dir_val: Directory containing validation embeddings for the small model (generated by precompute_logits.py)
        precomputed_large_emb_dir_val: Directory containing validation embeddings for the large model (generated by precompute_logits.py)
        precomputed_head_input_emb_dir_val: Directory containing validation embeddings to pass as input to the head, if different from precomputed_small_emb_dir (generated by precompute_logits.py)
        dataset_filter_path: Path to a dataset filter (generated by create_binned_dataset.py)
        val_dataset_filter_path: Path to a dataset filter (generated by create_binned_dataset.py)
        eval_every_n_batches: How often validation is performed (in batches)
        use_wandb: Whether to upload logs to Weights and Biases
        wandb_project: Weights and Biases project name. Mandatory with use_wandb.
        wandb_entity: Weights and Biases entity name. Mandatory with use_wandb.
        wandb_run_name: Weights and Biases run name. Mandatory with use_wandb.
    """
    # MUST COME FIRST
    args = locals()

    torch.manual_seed(seed)
    random.seed(seed)

    _validate_args(args)

    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)

    if(use_wandb):
        # Add some metrics that depend on arguments
        wandb_metrics.add(
            (f"val_confusion_matrix_{no_bins}", "step")
        )
        for i in range(no_bins):
            for j in range(no_bins):
                wandb_metrics.add(
                    (f"val_confusion_matrix_{no_bins}_{i}_{j}", "step")
                )

        # Init WandB, register metrics, etc.
        _wandb_setup(args)

    # Load the (small) LM heads of both the small and the large model.
    # We've only cached the embeddings and not the (much larger) logits.
    small_lm_head = load_lm_head(small_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=small_model_size, revision=small_model_revision)
    large_lm_head = load_lm_head(large_checkpoint_path, dtype=DTYPE, device=DEVICE, model_type=model_type, model_size=large_model_size)

    small_embedding_layer = None
    if(append_predicted_token_embedding):
        # Yes you could load this at the same time as the LM head above but who cares
        small_embedding_layer = load_embedding_layer(
            small_checkpoint_path, 
            dtype=DTYPE, 
            device=DEVICE, 
            model_type=model_type, 
            model_size=small_model_size, 
            revision=small_model_revision,
        )

    # Load the precomputed logits
    shard_dirs = [
        precomputed_small_emb_dir,
        precomputed_large_emb_dir,
    ]

    if(precomputed_head_input_emb_dir):
        shard_dirs.append(precomputed_head_input_emb_dir)

    logit_loader = PrecomputedShardLoader(
        shard_dirs, dataset_filter_path=dataset_filter_path
    )

    val = precomputed_small_emb_dir_val is not None
    if(val and precomputed_large_emb_dir_val is None):
        raise ValueError("Must provide both small and large validation directories")

    if(val):
        logging.info("Validation enabled...")
    else:
        logging.warning("Validation disabled...")

    val_logit_loader = None
    if(val):
        val_shard_dirs = [
            precomputed_small_emb_dir_val,
            precomputed_large_emb_dir_val,
        ]
        
        if(precomputed_head_input_emb_dir):
            assert(precomputed_head_input_emb_dir_val is not None)
            val_shard_dirs.append(precomputed_head_input_emb_dir_val)

        val_logit_loader = PrecomputedShardLoader(
            val_shard_dirs, dataset_filter_path=val_dataset_filter_path
        )

    # Initialize the model
    shared_head_params = {
        "no_bins": no_bins if bin_target else 1,
        "hidden_dim": hidden_dim,
        "no_hidden_layers": no_hidden_layers,
        "dropout": dropout,
        "activation": activation,
    }
    if(not glue_lm_head):
        if(use_logits_as_input):
            input_dim = small_lm_head.weight.shape[0]
        else:
            input_dim = small_lm_head.weight.shape[1]

        if(provide_entropy_as_input):
            input_dim += 1

        if(append_predicted_token_embedding):
            input_dim += small_embedding_layer.weight.shape[1]

        distance_prediction_head = DistancePredictionHead(
            input_dim=input_dim,
            **shared_head_params,
        )
    else:
        assert(not provide_entropy_as_input)
        assert(not append_predicted_token_embedding)
        distance_prediction_head = DistancePredictionHeadWithLMHead(
            lm_head=small_lm_head,
            **shared_head_params,
        )

    distance_prediction_head.to(DEVICE)
    param_count = sum(
        p.numel() for p in distance_prediction_head.parameters() if p.requires_grad
    )
    logging.info(f"Loaded prediction head ({param_count} parameters)...")

    # Umm uhh
    #distance_prediction_head = torch.compile(distance_prediction_head)

    if(resume_from):
        distance_prediction_head.load_state_dict(torch.load(resume_from, map_location=DEVICE))
        logging.info(f"Loaded model from {resume_from}...")

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        distance_prediction_head.parameters(), 
        lr=lr
    )

    # Select the loss function
    if(bin_target):
        loss_fn = lambda inputs, targets: torch.nn.functional.cross_entropy(
            inputs, targets.long()
        )
    else:
        if(le_loss_term):
            assert(target_fn_name == "large_entropy")
            loss_fn = le_loss_term_loss
        elif(underestimate_loss_term):
            assert(target_fn_name == "large_entropy")
            loss_fn = underestimate_loss_term_loss
        else:
            # Remember to remove the vestigial "bin" dimension
            loss_fn = lambda inputs, targets: torch.nn.functional.mse_loss(
                inputs.squeeze(-1), targets.to(dtype=DTYPE)
            )

    # Standard training loop
    cum_step = 0
    best_val_loss = float("inf")
    for epoch in range(no_epochs):
        if(use_wandb):
            wandb.log({"epoch": epoch})

        shuffle_seed = random.randint(0, 2**32 - 1)
        logit_loader.shuffle_shards(shuffle_seed)

        data_gen = _preprocessor(
            shard_loader=logit_loader,
            small_lm_head=small_lm_head,
            large_lm_head=large_lm_head,
            model_type=model_type,
            no_bins=no_bins,
            min_bin=min_bin,
            max_bin=max_bin,
            min_entropy=min_entropy,
            max_entropy=max_entropy,
            provide_entropy_as_input=provide_entropy_as_input,
            use_logits_as_input=use_logits_as_input,
            softmax_input_logits=softmax_input_logits,
            target_fn_name=target_fn_name,
            bin_target=bin_target,
            append_predicted_token_embedding=append_predicted_token_embedding,
            small_embedding_layer=small_embedding_layer,
            upsample_low_large_entropy=upsample_low_large_entropy,
            device=DEVICE,
            dtype=DTYPE,
        )

        bl = batch_loader(
            data_gen=data_gen,
            batch_size=batch_size,
            skip_frac=skip_frac,
            nonzero_bin_weight=nonzero_bin_weight,
        )

        for i, (inputs, targets) in enumerate(bl):
            # Dry run w/ grad enabled for the torch compiler (idk why this is necessary)
            if(i == 0 and epoch == 0):
                distance_prediction_head(inputs)

            # Periodically run the validation loop
            if(val and i % eval_every_n_batches == 0):
                val_stash = {}
                val_data_gen = _preprocessor(
                    shard_loader=val_logit_loader,
                    small_lm_head=small_lm_head,
                    large_lm_head=large_lm_head,
                    model_type=model_type,
                    no_bins=no_bins,
                    min_bin=min_bin,
                    max_bin=max_bin,
                    min_entropy=min_entropy,
                    max_entropy=max_entropy,
                    provide_entropy_as_input=provide_entropy_as_input,
                    use_logits_as_input=use_logits_as_input,
                    softmax_input_logits=softmax_input_logits,
                    target_fn_name=target_fn_name,
                    bin_target=bin_target,
                    append_predicted_token_embedding=append_predicted_token_embedding,
                    small_embedding_layer=small_embedding_layer,
                    device=DEVICE,
                    dtype=DTYPE,
                    _stash=val_stash, # used to smuggle out the entropy
                )

                val_bl = batch_loader(
                    data_gen=val_data_gen,
                    batch_size=batch_size,
                    skip_frac=0.,
                )

                with torch.no_grad():
                    val_loss_sum = 0
                    val_unweighted_loss_sum = 0
                    val_acc_sum = 0
                    val_batch_count = 0
                    all_val_preds = []
                    all_val_gt = []

                    distance_prediction_head.eval()

                    for j, (val_inputs, val_targets) in enumerate(val_bl):
                        val_inputs = val_inputs.to(DEVICE)
                        val_targets = val_targets.to(DEVICE)

                        val_inputs = val_inputs.to(torch.float32)

                        val_outputs = distance_prediction_head(val_inputs)
                        val_loss = loss_fn(val_outputs, val_targets)

                        if(le_loss_term or underestimate_loss_term):
                            val_unweighted_loss = torch.mean((val_outputs.squeeze(-1) - val_targets.to(dtype=DTYPE)) ** 2)
                        else:
                            val_unweighted_loss = val_loss

                        val_loss_sum += torch.sum(val_loss).item()
                        val_unweighted_loss_sum += torch.sum(val_unweighted_loss).item()

                        if(bin_target):
                            val_preds = torch.argmax(val_outputs, dim=-1)
                            val_acc = val_preds == val_targets
                            val_acc_sum += torch.mean(val_acc.float()).item()
                        else:
                            val_preds = val_outputs

                        val_batch_count += 1

                        all_val_preds.extend(val_preds.squeeze(-1).cpu().tolist())
                        all_val_gt.extend(val_targets.cpu().tolist())

                    # Compute the accuracy of the simplest entropy threshold
                    entropy_threshold_dict = {}
                    if(bin_target and target_fn_name == "large_entropy"):
                        entropy_threshold_dict["val_entropy_threshold_acc"] = (
                            entropy_threshold_acc(val_stash["small_entropy"], all_val_gt)
                        )
                    elif(target_fn_name == "large_entropy"):
                        entropy_threshold_dict["val_small_entropy_loss"] = (
                            loss_fn(
                                torch.tensor(val_stash["small_entropy"]).to(DTYPE),
                                torch.tensor(all_val_gt).to(DTYPE),
                            ).item()
                        )

                    mean_val_loss = val_loss_sum / val_batch_count
                    mean_val_unweighted_loss = val_unweighted_loss_sum / val_batch_count
                    val_metrics = {
                        "val_loss": mean_val_loss,
                        "val_unweighted_loss": mean_val_unweighted_loss,
                        "val_set_size": len(all_val_gt),
                        **entropy_threshold_dict,
                    }

                    print(f"Validation metric val_loss: {val_metrics['val_loss']}")

                    if(bin_target): # Classification
                        confusion_matrix = torch.zeros(no_bins, no_bins)
                        for gt, pred in zip(all_val_gt, all_val_preds):
                            confusion_matrix[gt, pred] += 1

                        confusion_matrix = confusion_matrix / (confusion_matrix.sum() + 1e-6)

                        val_metrics.update({
                            "val_accuracy": val_acc_sum / val_batch_count,
                            f"val_confusion_matrix_{no_bins}": confusion_matrix,
                        })

                        print(f"Validation metric val_accuracy: {val_metrics['val_accuracy']}")

                        if(use_wandb):
                            # Make a nice WandB confusion matrix
                            val_metrics[f"val_confusion_matrix_{no_bins}"] = wandb.plot.confusion_matrix(
                                y_true=all_val_gt,
                                preds=all_val_preds,
                                class_names=[str(label) for label in range(no_bins)],
                            )

                            # Annoyingly, WandB doesn't support plotting confusion matrices over time
                            # We need to add those values separately
                            for row in range(no_bins):
                                for col in range(no_bins):
                                    val_metrics[f"val_confusion_matrix_{no_bins}_{row}_{col}"] = (
                                        confusion_matrix[row,col]
                                    )
                    else: # Regression
                        if(target_fn_name == "large_entropy"):
                            for le in ENTROPY_THRESHOLDS_FOR_METRICS:
                                gt_below_le = [t for t in zip(all_val_preds, all_val_gt) if t[1] < le]
                                gt_below_le_mse = np.mean([(p - g) ** 2 for p, g in gt_below_le])
                                val_metrics[f"val_mse_below_{le}"] = gt_below_le_mse

                    if(use_wandb):
                        _wandb_log(metrics=val_metrics, step_metric="step", step=cum_step)

                    # Save the model
                    if(mean_val_loss < best_val_loss):
                        model_path = os.path.join(output_dir, "state_dict.pth")
                        torch.save(distance_prediction_head.state_dict(), model_path)
                        best_val_loss = mean_val_loss

                    distance_prediction_head.train()

            inputs = inputs.to(device=DEVICE, dtype=DTYPE)
            # Hold off on cast until the loss fn
            targets = targets.to(device=DEVICE)

            optimizer.zero_grad()
            outputs = distance_prediction_head(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            metrics = {
                "train_loss": loss.item(),
            }

            if(bin_target):
                accuracy = torch.sum(
                    torch.argmax(outputs, dim=-1) == targets
                ) / targets.numel()

                metrics.update({
                    "train_accuracy": accuracy.item(),
                })

            if(use_wandb):
                _wandb_log(metrics=metrics, step_metric="step", step=cum_step)

            if i % 100 == 0:
                print(f"Epoch {epoch}, batch {i}, loss: {loss.item():.02f}", file=sys.stderr)
                if(bin_target):
                    print(f"Epoch {epoch}, batch {i}, accuracy: {accuracy.item():.02f}", file=sys.stderr)

            cum_step += 1


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
