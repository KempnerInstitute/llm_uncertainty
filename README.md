# Distinguishing the Knowable from the Unknowable with Language Models

Code for the preprint "Distinguishing the Knowable from the Unknowable with Language Models," submitted to ICML 2024.

**Abstract**: We study the feasibility of identifying *epistemic* uncertainty (reflecting a lack of knowledge), as opposed to *aleatoric* uncertainty (reflecting entropy in the underlying distribution), in the outputs of large language models (LLMs) over free-form text. In the absence of ground-truth probabilities, we explore a setting where, in order to (approximately) disentangle a given LLM's uncertainty, a significantly larger model stands in as a proxy for the ground truth. We show that small linear probes trained on the embeddings of frozen, pretrained models accurately predict when larger models will be confident at the token level and that probes trained on one text domain generalize to others. Going further, we propose a fully unsupervised method that achieves non-trivial accuracy on the same task. Taken together, we interpret these results as evidence that LLMs naturally contain internal representations of different types of uncertainty that could potentially be leveraged to devise more informative indicators of model confidence in diverse practical settings.

## Instructions

### Obtaining model weights

Our scripts support (with some gaps) three models: LLaMA, Pythia, and Llama 2.

LLaMA: Follow the instructions in the original `lit-llama` repository [here](https://github.com/Lightning-AI/lit-llama/blob/main/howto/download_weights.md).
Pythia: These should download automatically from Huggingface upon first use
Llama 2: Follow the instructions in the `lit-gpt` repository [here](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_llama_2.md).


### Supervised experiments

First, precompute model embeddings with `precompute_logits.py`. Embeddings from different layers can be selected with the `return_after_layer_n` flag. We include our Wikipedia dataset (new Wikipedia articles after the LLaMA training cutoff) under `jsons/`.

Next, generate a dataset filter for a specific entropy band using `create_dataset_filter.py` (or `create_dataset_filter_no_gap.py` for a gapless filter).

Finally, train probes using `train_head.py.`

### Unsupervised experiments

ICLT is implemented in `repetition.py` and `generate_from_repetition.py`.
