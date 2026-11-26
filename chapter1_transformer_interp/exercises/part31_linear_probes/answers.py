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
    print(f"\n{name}: {len(df)} statements ({df['label'].sum()} true, {(1 - df['label']).sum():.0f} false)")
    display(df.head(4))
