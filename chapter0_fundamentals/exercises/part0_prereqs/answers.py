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


# %%
# Exercise - einops operations & broadcasting
def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")


# %%
# Exercise - einops operations & broadcasting A1
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    tensor = t.arange(start=3, end=9)
    tensor = einops.rearrange(tensor, "(w1 w2) -> w1 w2", w1=3)

    print(tensor)
    return tensor


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)


# %%
# Exercise - einops operations & broadcasting A2
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    tensor = einops.rearrange(t.arange(1, 7), "(h w) -> h w", h=2)
    print(tensor)
    return tensor


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))


# %%
# Exercise - einops operations & broadcasting B1
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    temps = einops.reduce(temps, "(w w1) -> w", "mean", w=2)
    print(temps)
    return temps


temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))


# %%
# Exercise - einops operations & broadcasting B2
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    avg_temps = einops.reduce(temps, "(w w1) -> w", "mean", w=2)
    avg_temps = einops.repeat(avg_temps, "w -> (w 7)")
    print(temps - avg_temps)
    return temps - avg_temps


expected = [
    -0.571,
    0.429,
    -1.571,
    3.429,
    -0.571,
    0.429,
    -1.571,
    -4.0,
    1.0,
    6.0,
    1.0,
    -1.0,
    -7.0,
    4.0,
]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))


# %%
# Exercise - einops operations & broadcasting B3
def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    avg_temps = einops.reduce(temps, "(w w1) -> w", "mean", w=2)
    avg_temps = einops.repeat(avg_temps, "w -> (w 7)")
    std_temps = einops.reduce(temps, "(w w1) -> w", t.std, w=2)
    std_temps = einops.repeat(std_temps, "w -> (w 7)")
    temps = (temps - avg_temps) / std_temps
    print(temps)
    return temps


expected = [
    -0.333,
    0.249,
    -0.915,
    1.995,
    -0.333,
    0.249,
    -0.915,
    -0.894,
    0.224,
    1.342,
    0.224,
    -0.224,
    -1.565,
    0.894,
]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))


# %%
# Exericse - C1
def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    norm = t.norm(matrix, dim=1)
    norm = einops.repeat(norm, "h -> h 3")
    matrix = matrix / norm
    print(matrix)
    return matrix


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor(
    [[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]]
)
assert_all_close(normalize_rows(matrix), expected)


# %%
# Exercise C2
def cos_sim_matrix(matrix: Tensor) -> Tensor:
    """Return the cosine similarity matrix for each pair of rows of the given matrix.

    matrix: shape (m, n)
    """
    norm = t.norm(matrix, dim=1)
    norm = einops.repeat(norm, "h -> h 3")
    matrix = matrix/norm
    matrix = t.matmul(matrix, matrix.T)
    print(matrix)
    return matrix


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
assert_all_close(cos_sim_matrix(matrix), expected)
