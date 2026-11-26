# %%
import arena_lib

exercises_dir, section_dir = arena_lib.setup("1.3.1")
# %%
import gc
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import circuitsvis as cv
import einops
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from datasets import load_dataset
from huggingface_hub import snapshot_download
from IPython.display import HTML, display
from jaxtyping import Bool, Float, Int
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Grow GPU memory blocks instead of caching many fixed-size ones: less fragmentation, so the day fits a 16GB GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# LOW_GPU_MEM fires automatically on GPUs with less than 30 GiB (e.g. Colab's 16GB T4), or with no CUDA GPU at all.
LOW_GPU_MEM = (
    not t.cuda.is_available()
    or t.cuda.get_device_properties(0).total_memory / (1024**3) < 30
)

import part31_linear_probes.tests as tests
import part31_linear_probes.utils as utils
from arena_lib.huggingface import get_act_name, run_with_cache
from arena_lib.quant import half_dtype, silence_int8_cast_warning

dtype = (
    half_dtype()
)  # bfloat16, or float16 (with a warning) on GPUs without native bfloat16, like Colab's T4
silence_int8_cast_warning()  # 8-bit layers log a line per forward pass when the model is bfloat16

MAIN = __name__ == "__main__"

# %%
# HF mirrors of the files we use from each repo (same paths as the originals)
GOT_ROOT = Path(
    snapshot_download("arena-education/geometry-of-truth", repo_type="dataset")
)
DD_ROOT = Path(
    snapshot_download("arena-education/deception-detection", repo_type="dataset")
)

GOT_DATASETS = GOT_ROOT / "datasets"
DD_DATA = DD_ROOT / "data"
# %%
HF_TOKEN = os.getenv("HF_TOKEN")
assert (
    HF_TOKEN
), "HF_TOKEN isn't set: add it (see above), then re-run the setup cell at the top"

# %%
MODEL_NAME = "meta-llama/Llama-2-13b-hf"

print(f"{LOW_GPU_MEM=}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map=(
        "cuda" if LOW_GPU_MEM else "auto"
    ),  # on 16GB, "auto" would offload some of the 8-bit model to the CPU
    # 26GB in bf16; threshold 2 (default 6) keeps more outliers in 16-bit, so results stay closer to bf16
    quantization_config=(
        BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=2.0)
        if LOW_GPU_MEM
        else None
    ),
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

NUM_LAYERS = len(model.model.layers)
D_MODEL = model.config.hidden_size
# Layer choices from the geometry-of-truth repo config for llama-2-13b. The paper
# found truth representations are concentrated in early-to-mid layers, and identified
# these specific layers via patching experiments (Section 3, "group (b)").
PROBE_LAYER = 14
INTERVENE_LAYER = 8

print(f"Model: {MODEL_NAME}")
print(f"Layers: {NUM_LAYERS}, Hidden dim: {D_MODEL}")
print(f"Probe layer: {PROBE_LAYER}, Intervene layer: {INTERVENE_LAYER}")

# %%
DATASET_NAMES = ["cities", "sp_en_trans", "larger_than"]

datasets = {}
for name in DATASET_NAMES:
    df = pd.read_csv(GOT_DATASETS / f"{name}.csv")
    datasets[name] = df
    print(
        f"\n{name}: {len(df)} statements ({df['label'].sum()} true, {(1 - df['label']).sum():.0f} false)"
    )
    display(df.head(4))


# %%
def extract_activations(
    statements: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    batch_size: int = 25,
) -> dict[int, Float[Tensor, "n_statements d_model"]]:
    """
    Extract last-token hidden state activations from specified layers for a list of statements.

    Args:
        statements: List of text statements to process.
        model: A HuggingFace causal language model.
        tokenizer: The corresponding tokenizer.
        layers: List of layer indices (0-indexed) to extract activations from.
        batch_size: Number of statements to process at once.

    Returns:
        Dictionary mapping layer index to tensor of activations (on CPU, float32), shape [n_statements, d_model].
    """
    names = [get_act_name("resid_post", layer) for layer in layers]
    activations = {layer: [] for layer in layers}
    for i in range(0, len(statements), batch_size):
        inputs = tokenizer(
            statements[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        with t.no_grad():
            _, cache = run_with_cache(
                model,
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                names_filter=names,
                stop_at_layer=max(layers) + 1,
            )
        last_indices = inputs["attention_mask"].sum(dim=1) - 1
        for layer in layers:
            activations[layer].append(
                cache["resid_post", layer][
                    t.arange(len(last_indices), device=last_indices.device),
                    last_indices,
                    :,
                ]
                .cpu()
                .float()
            )
    for layer in layers:
        activations[layer] = t.cat(activations[layer])
    return activations


tests.test_extract_activations(
    extract_activations, model, tokenizer, PROBE_LAYER, D_MODEL
)
# %%
# Extract activations at the probe layer for all datasets
activations = {}
labels_dict = {}
statements_dict = {}

for name in DATASET_NAMES:
    df = datasets[name]
    statements = df["statement"].tolist()
    labs = t.tensor(df["label"].values, dtype=t.float32)
    statements_dict[name] = statements

    acts = extract_activations(statements, model, tokenizer, [PROBE_LAYER])
    activations[name] = acts[PROBE_LAYER]
    labels_dict[name] = labs

# Show summary table
summary = pd.DataFrame(
    {
        "Dataset": DATASET_NAMES,
        "N statements": [len(datasets[n]) for n in DATASET_NAMES],
        "N true": [int(datasets[n]["label"].sum()) for n in DATASET_NAMES],
        "N false": [int((1 - datasets[n]["label"]).sum()) for n in DATASET_NAMES],
        "Act shape": [str(tuple(activations[n].shape)) for n in DATASET_NAMES],
        "Mean norm": [f"{activations[n].norm(dim=-1).mean():.1f}" for n in DATASET_NAMES],
    }
)
display(summary)

# %%
def get_pca_components(
    activations: Float[Tensor, "n d_model"],
    k: int = 2,
) -> Float[Tensor, "d_model k"]:
    """
    Compute the top-k principal components of the activation matrix.

    Args:
        activations: Activation matrix, shape [n_samples, d_model].
        k: Number of principal components to return.

    Returns:
        Matrix of top-k eigenvectors as columns, shape [d_model, k].
    """
    raise NotImplementedError()


tests.test_get_pca_components(get_pca_components, activations["cities"], D_MODEL)
