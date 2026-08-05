# %%
import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from plotly_utils import line

import gc, sys


def cuda_reset():
    for attr in ("last_traceback", "last_value", "last_type", "last_exc"):
        if hasattr(sys, attr):
            setattr(sys, attr, None)
    ip = get_ipython()  # type: ignore
    if ip is not None:
        ip.run_line_magic("reset", "-f out")  # clears Out[]/_ /__ history
    gc.collect()
    t.cuda.empty_cache()
    print(
        f"{t.cuda.memory_allocated()/1e9:.2f} GB allocated, "
        f"{t.cuda.memory_reserved()/1e9:.2f} GB reserved"
    )


# %%
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        lower_than_zero = x < 0
        x[lower_than_zero] = 0
        return x


tests.test_relu(ReLU)


# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        sqrt_in = np.sqrt(in_features)
        self.weight = nn.Parameter(
            2 * t.rand([out_features, in_features]) / sqrt_in - 1 / sqrt_in
        )
        if bias:
            self.bias = nn.Parameter(2 * t.rand(out_features) / sqrt_in - 1 / sqrt_in)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        # Original ugly solution
        # n = x.shape[0]
        # if self.bias is not None:
        #     x = t.concat([x, t.ones([n, 1])], dim=1)
        #     assert x.shape == (n, self.in_features + 1)
        #     mat = t.concat([self.weight, self.bias.unsqueeze(-1)], dim=1)
        #     assert mat.shape == (self.out_features, self.in_features + 1)
        # else:
        #     mat = self.weight
        # return x @ mat.T
        x = einops.einsum(x, self.weight, "... in_feats, out_feats in_feats -> ... out_feats")
        if self.bias is not None:
            x += self.bias
        return x

    def extra_repr(self) -> str:
        return f"Weights shape {self.weight.shape}\n Bias shape {self.bias.shape if self.bias is not None else None}"


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)
