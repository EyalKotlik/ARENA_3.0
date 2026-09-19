# %%
import functools
import sys
from pathlib import Path
from typing import Callable

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import circuitsvis as cv
import einops
import numpy as np
import torch as t
import torch.nn as nn
from eindex import eindex
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
    utils,
)
from transformer_lens.hook_points import HookPoint

device = t.device(
    "mps"
    if t.backends.mps.is_available()
    else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part2_intro_to_mech_interp"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_intro_to_mech_interp.tests as tests
from plotly_utils import (
    hist,
    imshow,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

MAIN = __name__ == "__main__"
# %%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# %%
print(gpt2_small.cfg)
# %%
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly.

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)
# %%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_str_tokens(["gpt2", "gpt2"]))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))
# %%
logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]

# YOUR CODE HERE - get the model's prediction on the text
input_tokens = gpt2_small.to_tokens(model_description_text).squeeze()[1:]
correct_predictions = prediction == input_tokens
print(f"Correct predictions {correct_predictions.sum()}/{len(input_tokens)}")
print(gpt2_small.to_str_tokens(prediction[correct_predictions]))
# %%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on task-specific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

print(type(gpt2_logits), type(gpt2_cache))
# %%
attn_patterns_from_shorthand = gpt2_cache["pattern", 0]
attn_patterns_from_full_name = gpt2_cache["blocks.0.attn.hook_pattern"]

t.testing.assert_close(attn_patterns_from_shorthand, attn_patterns_from_full_name)
# %%
layer0_pattern_from_cache = gpt2_cache["pattern", 0]

# YOUR CODE HERE - define `layer0_pattern_from_q_and_k` manually, by manually performing the
# steps of the attention calculation (dot product, masking, scaling, softmax)
layer0_q = gpt2_cache["q", 0]
layer0_k = gpt2_cache["k", 0]
print(layer0_q.shape)
print(layer0_k.shape)
layer0_attn_scores = einops.einsum(
    layer0_q, layer0_k, "sQ nhead dhead, sK nhead dhead -> nhead sQ sK"
)
layer0_attn_scores = layer0_attn_scores / (layer0_k.shape[-1] ** 0.5)
mask = t.ones_like(layer0_attn_scores).triu(diagonal=1) == 1
layer0_attn_scores[mask] = -t.inf
t.testing.assert_close(layer0_attn_scores, gpt2_cache["attn_scores", 0])
layer0_pattern_from_q_and_k = layer0_attn_scores.softmax(-1)

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")
# %%
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=gpt2_str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)
# %%
neuron_activations_for_all_layers = t.stack(
    [gpt2_cache["post", layer] for layer in range(gpt2_small.cfg.n_layers)], dim=1
)
# shape = (seq_pos, layers, neurons)

cv.activations.text_neuron_activations(
    tokens=gpt2_str_tokens, activations=neuron_activations_for_all_layers
)
# %%
neuron_activations_for_all_layers_rearranged = utils.to_numpy(
    einops.rearrange(
        neuron_activations_for_all_layers, "seq layers neurons -> 1 layers seq neurons"
    )
)

cv.topk_tokens.topk_tokens(
    # Some weird indexing required here ¯\_(ツ)_/¯
    tokens=[gpt2_str_tokens],
    activations=neuron_activations_for_all_layers_rearranged,
    max_k=7,
    first_dimension_name="Layer",
    third_dimension_name="Neuron",
    first_dimension_labels=list(range(12)),
)
# %%
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)
# %%
from huggingface_hub import hf_hub_download

REPO_ID = "callummcdougall/attn_only_2L_half"
FILENAME = "attn_only_2L_half.pth"

weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
# %%
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_path, map_location=device, weights_only=True)
model.load_state_dict(pretrained_weights)
# %%
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)
attention_pattern = []
attention_pattern.append(cache["pattern", 0])
attention_pattern.append(cache["pattern", 1])
tokens = model.to_str_tokens(text)
for i in range(len(attention_pattern)):
    print(f"Layer {i} attention pattern: ")
    display(
        cv.attention.attention_patterns(
            tokens=tokens,
            attention=attention_pattern[i],
            attention_head_names=[f"L{i}H{j}" for j in range(model.cfg.n_heads)],
        )
    )


# %%
def current_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    current_attn_heads = []
    for layer_idx in range(model.cfg.n_layers):
        attention = cache["pattern", layer_idx]
        for head_idx in range(model.cfg.n_heads):
            head = attention[head_idx]
            head_current_prob_mean = head.diagonal().mean()
            if head_current_prob_mean > 0.4:
                current_attn_heads.append(f"{layer_idx}.{head_idx}")
    return current_attn_heads


def prev_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    current_attn_heads = []
    for layer_idx in range(model.cfg.n_layers):
        attention = cache["pattern", layer_idx]
        for head_idx in range(model.cfg.n_heads):
            head = attention[head_idx]
            head_current_prob_mean = head.diagonal(-1).mean()
            if head_current_prob_mean > 0.4:
                current_attn_heads.append(f"{layer_idx}.{head_idx}")
    return current_attn_heads


def first_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    current_attn_heads = []
    for layer_idx in range(model.cfg.n_layers):
        attention = cache["pattern", layer_idx]
        for head_idx in range(model.cfg.n_heads):
            head = attention[head_idx]
            head_current_prob_mean = head[:, 0].mean()
            if head_current_prob_mean > 0.4:
                current_attn_heads.append(f"{layer_idx}.{head_idx}")
    return current_attn_heads


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> Int[Tensor, "batch_size full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
    """
    t.manual_seed(0)  # for reproducibility
    prefix = (t.ones(batch_size, 1) * model.tokenizer.bos_token_id).long()
    random_tokens = t.randint(0, model.cfg.d_vocab, (batch_size, seq_len))
    random_tokens = t.cat((prefix, random_tokens, random_tokens), dim=-1)
    assert random_tokens.shape == (batch_size, 1 + 2 * seq_len)

    return random_tokens


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch_size: int = 1
) -> tuple[Tensor, Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning (tokens,
    logits, cache). This function should use the `generate_repeated_tokens` function above.

    Outputs are:
        rep_tokens: [batch_size, 1+2*seq_len]
        rep_logits: [batch_size, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    random_tokens = generate_repeated_tokens(model, seq_len, batch_size)
    logits, cache = model.run_with_cache(random_tokens)
    return random_tokens, logits, cache


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    logprobs = logits.log_softmax(dim=-1)
    # We want to get logprobs[b, s, tokens[b, s+1]], in eindex syntax this looks like:
    correct_logprobs = eindex(logprobs, tokens, "b s [b s+1]")
    return correct_logprobs


seq_len = 50
batch_size = 1
rep_tokens, rep_logits, rep_cache = run_and_cache_model_repeated_tokens(
    model, seq_len, batch_size
)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)
# %%
for i in range(model.cfg.n_layers):
    attention_pattern = rep_cache["pattern", i]
    print(f"Layer {i} attention pattern: ")
    display(
        cv.attention.attention_patterns(
            tokens=rep_str,
            attention=attention_pattern,
            attention_head_names=[f"L{i}H{j}" for j in range(model.cfg.n_heads)],
        )
    )


# %%
def induction_attn_detector(cache: ActivationCache) -> list[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    heads = []
    for layer_idx in range(model.cfg.n_layers):
        attention = cache["pattern", layer_idx]
        for head_idx in range(model.cfg.n_heads):
            head = attention[head_idx]
            head_current_prob_mean = head.diagonal(-seq_len + 1).mean()
            if head_current_prob_mean > 0.4:
                heads.append(f"{layer_idx}.{head_idx}")
    return heads


print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))
# %%
seq_len = 50
batch_size = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch_size)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU,
# which can be slow.
induction_score_store = t.zeros(
    (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"], hook: HookPoint
):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the
    `induction_score_store` tensor.
    """
    induction_scores = (
        pattern.diagonal(offset=-(seq_len - 1), dim1=2, dim2=3).mean(dim=-1).mean(dim=0)
    )
    induction_score_store[hook.layer(), :] = induction_scores


# We make a boolean filter on activation names, that's true only on attention pattern names
pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
)


# %%
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), attention=pattern.mean(0)
        )
    )


# YOUR CODE HERE - find induction heads in gpt2_small
induction_score_store = t.zeros(
    (gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device
)
gpt2_small.run_with_hooks(
    rep_tokens_10,
    return_type=None,
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=350,
)

induction_head_layers = [5, 6, 7]
visualization_hooks = [
    (utils.get_act_name("pattern", layer), visualize_pattern_hook)
    for layer in induction_head_layers
]
gpt2_small.run_with_hooks(
    rep_tokens_10, return_type=None, fwd_hooks=visualization_hooks
)


# %%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dims)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dims)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]
    direct_path = einops.einsum(
        embed[:-1], W_U_correct_tokens, "seq d_model, d_model seq -> seq"
    )
    l1_path = einops.einsum(
        l1_results[:-1],
        W_U_correct_tokens,
        "seq nheads d_model, d_model seq -> seq nheads",
    )
    l2_path = einops.einsum(
        l2_results[:-1],
        W_U_correct_tokens,
        "seq nheads d_model, d_model seq -> seq nheads",
    )
    return t.concat((direct_path.unsqueeze(-1), l1_path, l2_path), dim=-1)


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")
# %%
embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(
    embed, l1_results, l2_results, model.W_U, tokens.squeeze()
)

plot_logit_attribution(
    model, logit_attr, tokens, title="Logit attribution (demo prompt)"
)
# %%
# YOUR CODE HERE - plot logit attribution for the induction sequence (i.e. using `rep_tokens` and
# `rep_cache`), and interpret the results.
embed = rep_cache["embed"]
l1_results = rep_cache["result", 0]
l2_results = rep_cache["result", 1]
logit_attr = logit_attribution(
    embed, l1_results, l2_results, model.W_U, rep_tokens.squeeze()
)
plot_logit_attribution(model, logit_attr, rep_tokens, title="Logit attribution 2")


# %%
def head_zero_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :].zero_()


def get_ablation_scores(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    ablation_function: Callable = head_zero_ablation_hook,
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss
    from ablating the output of each head.
    """
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros(
        (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
    )

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    seq_len = (tokens.shape[1] - 1) // 2
    logits = model(tokens, return_type="logits")
    loss_no_ablation = -get_log_probs(logits, tokens)[:, -(seq_len - 1) :].mean()

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            layer_head_hook = functools.partial(
                head_zero_ablation_hook, head_index_to_ablate=head
            )
            logits = model.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=[(utils.get_act_name("z", layer), layer_head_hook)],
            )
            ablation_scores[layer, head] = -get_log_probs(logits, tokens)[
                :, -(seq_len - 1) :
            ].mean()

    return ablation_scores - loss_no_ablation


ablation_scores = get_ablation_scores(model, rep_tokens)
tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)
# %%
imshow(
    ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Loss Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=350,
)


# %%
def head_mean_ablation_hook(
    z: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> None:
    z[:, :, head_index_to_ablate, :] = z[:, :, head_index_to_ablate, :].mean(0)


rep_tokens_batch = run_and_cache_model_repeated_tokens(
    model, seq_len=50, batch_size=10
)[0]
mean_ablation_scores = get_ablation_scores(
    model, rep_tokens_batch, ablation_function=head_mean_ablation_hook
)

imshow(
    mean_ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Loss Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=350,
)

# %%
A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(
    f"Right dim: {AB_factor.rdim}, Left dim: {AB_factor.ldim}, Hidden dim: {AB_factor.mdim}"
)
# %%
print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)

print("\nSingular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)

print("\nFull SVD:")
print(AB_factor.svd())
# %%
C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C

print(f"Unfactored: shape={ABC.shape}, norm={ABC.norm()}")
print(f"Factored: shape={ABC_factor.shape}, norm={ABC_factor.norm()}")
print(
    f"\nRight dim: {ABC_factor.rdim}, Left dim: {ABC_factor.ldim}, Hidden dim: {ABC_factor.mdim}"
)
# %%
AB_unfactored = AB_factor.AB
t.testing.assert_close(AB_unfactored, AB)
# %%
head_index = 4
layer = 1

# YOUR CODE HERE - complete the `full_OV_circuit` object
OV_circuit = FactoredMatrix(model.W_V[layer, head_index], model.W_O[layer, head_index])
full_OV_circuit = model.W_E @ OV_circuit @ model.W_U

tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)
# %%
indices = t.randint(0, model.cfg.d_vocab, (200,))
full_OV_circuit_sample = full_OV_circuit[indices, indices].AB

imshow(
    full_OV_circuit_sample,
    labels={"x": "Logits on output token", "y": "Input token"},
    title="Full OV circuit for copying head",
    width=700,
    height=600,
)