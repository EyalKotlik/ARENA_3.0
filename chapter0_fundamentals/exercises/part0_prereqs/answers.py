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
    matrix = matrix / norm
    matrix = t.matmul(matrix, matrix.T)
    print(matrix)
    return matrix


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[1.0, 0.975, 0.959], [0.975, 1.0, 0.998], [0.959, 0.998, 1.0]])
assert_all_close(cos_sim_matrix(matrix), expected)


# %%
# Exercise D
def sample_distribution(probs: Tensor, n: int) -> Tensor:
    """Return n random samples from probs, where probs is a normalized probability distribution.

    probs: shape (k,) where probs[i] is the probability of event i occurring.
    n: number of random samples

    Return: shape (n,) where out[i] is an integer indicating which event was sampled.

    Use t.rand and t.cumsum to do this without any explicit loops.
    """
    cum_probs = t.cumsum(probs, dim=0)
    sample = einops.repeat(t.rand(n), "w -> w 6")
    sample = einops.reduce(sample > cum_probs, " h w -> h", "sum")
    return sample


n = 5_000_000
probs = t.tensor([0.05, 0.1, 0.1, 0.2, 0.15, 0.4])
freqs = t.bincount(sample_distribution(probs, n)) / n
assert_all_close(freqs, probs)


# %%
# Exercise E
def classifier_accuracy(scores: Tensor, true_classes: Tensor) -> Tensor:
    """Return the fraction of inputs for which the maximum score corresponds to the true class for that input.

    scores: shape (batch, n_classes). A higher score[b, i] means that the classifier thinks class i is more likely.
    true_classes: shape (batch, ). true_classes[b] is an integer from [0...n_classes).

    Use t.argmax.
    """
    prediction = scores.argmax(dim=1)
    prediction = true_classes - prediction
    accuracy = (prediction == 0).sum()/scores.shape[0]
    print(accuracy)
    return accuracy


scores = t.tensor([[0.75, 0.5, 0.25], [0.1, 0.5, 0.4], [0.1, 0.7, 0.2]])
true_classes = t.tensor([0, 1, 0])
expected = 2.0 / 3.0
assert classifier_accuracy(scores, true_classes) == expected
print("Tests passed!")
# %%
# Exercise F1
def total_price_indexing(prices: Tensor, items: Tensor) -> float:
    """Given prices for each kind of item and a tensor of items purchased, return the total price.

    prices: shape (k, ). prices[i] is the price of the ith item.
    items: shape (n, ). A 1D tensor where each value is an item index from [0..k).

    Use integer array indexing. The below document describes this for NumPy but it's the same in PyTorch:

    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
    """
    price = prices[items].sum().item()
    print(price)
    return price

prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_indexing(prices, items) == 9.0
print("Tests passed!")
# %%
# Exercise F2
def gather_2d(matrix: Tensor, indexes: Tensor) -> Tensor:
    """Perform a gather operation along the second dimension.

    matrix: shape (m, n)
    indexes: shape (m, k)

    Return: shape (m, k). out[i][j] = matrix[i][indexes[i][j]]

    For this problem, the test already passes and it's your job to write at least three asserts relating the arguments and the output. This is a tricky function and worth spending some time to wrap your head around its behavior.

    See: https://pytorch.org/docs/stable/generated/torch.gather.html?highlight=gather#torch.gather
    """
    # YOUR CODE HERE - add assert statement(s) here for `indices` and `matrix`
    assert matrix.ndim == indexes.ndim
    assert matrix.shape[0] >= indexes.shape[0]
    out = matrix.gather(1, indexes)
    # YOUR CODE HERE - add assert statement(s) here for `out`
    assert out.shape == indexes.shape

    return out


matrix = t.arange(15).view(3, 5)
indexes = t.tensor([[4], [3], [2]])
expected = t.tensor([[4], [8], [12]])
assert_all_equal(gather_2d(matrix, indexes), expected)

indexes2 = t.tensor([[2, 4], [1, 3], [0, 2]])
expected2 = t.tensor([[2, 4], [6, 8], [10, 12]])
assert_all_equal(gather_2d(matrix, indexes2), expected2)
# %%
# Exercise F3
def total_price_gather(prices: Tensor, items: Tensor) -> float:
    """Compute the same as total_price_indexing, but use torch.gather."""
    assert items.max() < prices.shape[0]
    price = prices.gather(0,items).sum().item()
    return price


prices = t.tensor([0.5, 1, 1.5, 2, 2.5])
items = t.tensor([0, 0, 1, 1, 4, 3, 2])
assert total_price_gather(prices, items) == 9.0
print("Tests passed!")
# %%
# Exercise G
def integer_array_indexing(matrix: Tensor, coords: Tensor) -> Tensor:
    """Return the values at each coordinate using integer array indexing.

    For details on integer array indexing, see:
    https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing

    matrix: shape (d_0, d_1, ..., d_n)
    coords: shape (batch, n)

    Return: (batch, )
    """
    values = matrix[[coords[:,i] for i in range(coords.shape[1])]] # my shitty solution
    # actual solution matrix[tuple(coords.T)]
    return values

mat_2d = t.arange(15).view(3, 5)
coords_2d = t.tensor([[0, 1], [0, 4], [1, 4]])
actual = integer_array_indexing(mat_2d, coords_2d)
assert_all_equal(actual, t.tensor([1, 4, 9]))

mat_3d = t.arange(2 * 3 * 4).view((2, 3, 4))
coords_3d = t.tensor([[0, 0, 0], [0, 1, 1], [0, 2, 2], [1, 0, 3], [1, 2, 0]])
actual = integer_array_indexing(mat_3d, coords_3d)
assert_all_equal(actual, t.tensor([0, 5, 10, 15, 20]))
# %%
# Exercise H1
def batched_logsumexp(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute log(sum(exp(row))) in a numerically stable way.

    matrix: shape (batch, n)

    Return: (batch, ). For each i, out[i] = log(sum(exp(matrix[i]))).

    Do this without using PyTorch's logsumexp function.

    A couple useful blogs about this function:
    - https://leimao.github.io/blog/LogSumExp/
    - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    """
    a=matrix.max(dim=-1).values
    exps = t.exp(matrix - einops.rearrange(a, "n -> n 1"))
    return a + t.log(t.sum(exps,dim=-1))

matrix = t.tensor([[-1000, -1000, -1000, -1000], [1000, 1000, 1000, 1000]])
expected = t.tensor([-1000 + math.log(4), 1000 + math.log(4)])
actual = batched_logsumexp(matrix)
assert_all_close(actual, expected)

matrix2 = t.randn((10, 20))
expected2 = t.logsumexp(matrix2, dim=-1)
actual2 = batched_logsumexp(matrix2)
assert_all_close(actual2, expected2)
# %%
# Exercise H2
def batched_softmax(matrix: Tensor) -> Tensor:
    """For each row of the matrix, compute softmax(row).

    Do this without using PyTorch's softmax function.
    Instead, use the definition of softmax: https://en.wikipedia.org/wiki/Softmax_function

    matrix: shape (batch, n)

    Return: (batch, n). For each i, out[i] should sum to 1.
    """
    #matrix = t.tensor([[1,1,1],[2,2,2]])
    # matrix = matrix.exp()
    # row_sums = einops.reduce(matrix, "h w -> h","sum")
    # row_sums = einops.repeat(row_sums, "n -> n batch", batch=matrix.shape[1])
    # result = matrix/row_sums
    # return result
    exp = matrix.exp()
    return exp / exp.sum(dim=-1, keepdim=True)
    raise NotImplementedError()


matrix = t.arange(1, 6).view((1, 5)).float().log()
expected = t.arange(1, 6).view((1, 5)) / 15.0
actual = batched_softmax(matrix)
assert_all_close(actual, expected)
for i in [0.12, 3.4, -5, 6.7]:
    assert_all_close(actual, batched_softmax(matrix + i))  # check it's translation-invariant

matrix2 = t.rand((10, 20))
actual2 = batched_softmax(matrix2)
assert actual2.min() >= 0.0
assert actual2.max() <= 1.0
assert_all_equal(actual2.argsort(), matrix2.argsort())
assert_all_close(actual2.sum(dim=-1), t.ones(matrix2.shape[:-1]))
# %%
# Exercise H3
def batched_logsoftmax(matrix: Tensor) -> Tensor:
    """Compute log(softmax(row)) for each row of the matrix.

    matrix: shape (batch, n)

    Return: (batch, n).

    Do this without using PyTorch's logsoftmax function.
    For each row, subtract the maximum first to avoid overflow if the row contains large values.
    """
    matrix = matrix - matrix.max(dim=-1,keepdim=True).values
    matrix = matrix - matrix.exp().sum(dim=1,keepdim=True).log()
    return matrix


matrix = t.arange(1, 7).view((2, 3)).float()
start = 1000
matrix2 = t.arange(start + 1, start + 7).view((2, 3)).float()
actual = batched_logsoftmax(matrix2)
expected = t.tensor([[-2.4076, -1.4076, -0.4076], [-2.4076, -1.4076, -0.4076]])
assert_all_close(actual, expected)
# %%
# Exercise H4
def batched_cross_entropy_loss(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Compute the cross entropy loss for each example in the batch.

    logits: shape (batch, classes). logits[i][j] is the unnormalized prediction for example i and class j.
    true_labels: shape (batch, ). true_labels[i] is an integer index representing the true class for example i.

    Return: shape (batch, ). out[i] is the loss for example i.

    Hint: convert the logits to log-probabilities using your batched_logsoftmax from above.
    Then the loss for an example is just the negative of the log-probability that the model assigned to the true class. Use torch.gather to perform the indexing.
    """
    assert logits.shape[0] == true_labels.shape[0]
    assert true_labels.max() < logits.shape[1]
    log_probs = batched_logsoftmax(logits)
    ce_loss = -log_probs[t.arange(logits.shape[1]),true_labels]
    return ce_loss
    raise NotImplementedError()


logits = t.tensor([[float("-inf"), float("-inf"), 0], [1 / 3, 1 / 3, 1 / 3], [float("-inf"), 0, 0]])
true_labels = t.tensor([2, 0, 0])
expected = t.tensor([0.0, math.log(3), float("inf")])
actual = batched_cross_entropy_loss(logits, true_labels)
assert_all_close(actual, expected)
# %%
# Exercise I1
def collect_rows(matrix: Tensor, row_indexes: Tensor) -> Tensor:
    """Return a 2D matrix whose rows are taken from the input matrix in order according to row_indexes.

    matrix: shape (m, n)
    row_indexes: shape (k,). Each value is an integer in [0..m).

    Return: shape (k, n). out[i] is matrix[row_indexes[i]].
    """
    return matrix[row_indexes]
    raise NotImplementedError()


matrix = t.arange(15).view((5, 3))
row_indexes = t.tensor([0, 2, 1, 0])
actual = collect_rows(matrix, row_indexes)
expected = t.tensor([[0, 1, 2], [6, 7, 8], [3, 4, 5], [0, 1, 2]])
assert_all_equal(actual, expected)
# %%
# Exercise I2
def collect_columns(matrix: Tensor, column_indexes: Tensor) -> Tensor:
    """Return a 2D matrix whose columns are taken from the input matrix in order according to column_indexes.

    matrix: shape (m, n)
    column_indexes: shape (k,). Each value is an integer in [0..n).

    Return: shape (m, k). out[:, i] is matrix[:, column_indexes[i]].
    """
    assert column_indexes.max() < matrix.shape[1]
    return matrix.T[column_indexes].T
    raise NotImplementedError()


matrix = t.arange(15).view((5, 3))
column_indexes = t.tensor([0, 2, 1, 0])
actual = collect_columns(matrix, column_indexes)
expected = t.tensor([[0, 2, 1, 0], [3, 5, 4, 3], [6, 8, 7, 6], [9, 11, 10, 9], [12, 14, 13, 12]])
assert_all_equal(actual, expected)
# %%
# Exercises - Einsum
def einsum_trace(mat: np.ndarray):
    """
    Returns the same as `np.trace`.
    """
    diag = np.identity(mat.shape[0])
    return einops.einsum(mat, diag, "i j, i j -> ")
    raise NotImplementedError()


def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    """
    return einops.einsum(mat, vec, "i j, j -> i")
    raise NotImplementedError()


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    """
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    """
    return einops.einsum(mat1, mat2, "m n, n k ->m k")
    raise NotImplementedError()


def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.inner`.
    """
    return einops.einsum(vec1, vec2, "i, i -> ")
    raise NotImplementedError()


def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    """
    Returns the same as `np.outer`.
    """
    return einops.einsum(vec1, vec2, "i, j -> i j")
    raise NotImplementedError()


tests.test_einsum_trace(einsum_trace)
tests.test_einsum_mv(einsum_mv)
tests.test_einsum_mm(einsum_mm)
tests.test_einsum_inner(einsum_inner)
tests.test_einsum_outer(einsum_outer)
