# %%
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"
# %%
# Einops
arr = np.load(section_dir / "numbers.npy")
print(arr.shape)

print(arr[0].shape)
display_array_as_img(arr[0])

print(arr[0, 0].shape)
display_array_as_img(arr[0, 0])

arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr_stacked.shape)
display_array_as_img(arr_stacked)
# %%
# Exercise - Einops Operations 1
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
print(arr1.shape)
display_array_as_img(arr1)
# %%
# Exercise - Einops Operations 2
arr2 = arr[0]
arr2 = einops.repeat(arr2, "c h w -> c (2 h) w")
display_array_as_img(arr2)
# %%
# Exercise - Einops Operations 3
arr3 = einops.repeat([arr[0], arr[1]], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)
# %%
# Exercise - Einops Operations 4
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(arr4)
# %%
# Exercise - Einops Operations 5
arr5 = einops.repeat(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)
# %%
# Exercise - Einops Operations 6
arr6 = einops.rearrange(arr[0:6], "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
display_array_as_img(arr6)
# %%
# Exercise - Einops Operations 7
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr7)
# %%
# Exercise - Einops Operations 8
arr8 = einops.reduce(
    arr[0:6], "(b1 b2) c (h1 h2) (w1 w2) -> c (b1 h1) (b2 w1)", "max", b1=2, h2=2, w2=2
)
display_array_as_img(arr8)
