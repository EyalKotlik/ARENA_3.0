# %%
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"


# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    # result = []
    # origin = [0, 0, 0]
    # result.extend([[origin, [1, j*2*y_limit/(num_pixels-1) -y_limit, 0]] for j in range(num_pixels)])
    # return t.tensor(result)
    result = t.zeros((num_pixels, 2, 3))
    result[:, 1, 0] = 1
    # result[:, 1, 1] = t.arange(-y_limit,y_limit+y_limit/(num_pixels-1),2*y_limit/(num_pixels-1))
    t.linspace(-y_limit, y_limit, num_pixels, out=result[:, 1, 1])
    print(result)
    return result


rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)


# %%
def intersect_ray_1d(
    ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]
) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    O = ray[0, 0:2]
    D = ray[1, 0:2]
    L1 = segment[0, 0:2]
    L2 = segment[1, 0:2]
    matrix = t.stack([D, L1 - L2], dim=1)
    try:
        solution = t.linalg.solve(matrix, L1 - O)
        return solution[0] >= 0 and 0 <= solution[1] <= 1
    except:
        return False


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)


# %%
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    rays = rays[..., 0:2]
    segments = segments[..., 0:2]
    O = rays[:, 0]
    D = rays[:, 1]
    L1 = segments[:, 0]
    L2 = segments[:, 1]
    L= L1-L2
    L = einops.repeat(L,"h w -> (nrays h) w", nrays=rays.shape[0])
    D = einops.repeat(D,"h w -> (h nsegments) w", nsegments=segments.shape[0])
    A = t.stack([D, L], dim=-1)
    L1 = einops.repeat(L1,"h w -> (nrays h) w", nrays=rays.shape[0])
    O = einops.repeat(O,"h w -> (h nsegments) w", nsegments=segments.shape[0])
    B = L1 - O
    determinants = t.linalg.det(A)
    determinants = determinants.abs() < 1e-8
    A[determinants] = t.eye(2)
    solution = t.linalg.solve(A,B)
    solution = (solution[:,0] >=0) & (solution[:,1]<= 1) & (0<=solution[:,1]) & ~determinants
    solution = einops.rearrange(solution, "(b1 b) -> b1 b", b=segments.shape[0])
    solution = solution.any(dim=-1)
    return solution
    raise NotImplementedError()


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    result = t.zeros((num_pixels_y,num_pixels_z,2, 3))
    result[:, :, 1, 0] = 1
    y_coords = t.linspace(-y_limit, y_limit, num_pixels_y)
    z_coords = t.linspace(-z_limit, z_limit, num_pixels_z)
    coords = t.cartesian_prod(y_coords,z_coords)
    coords = einops.rearrange(coords, "(b1 b) d-> b b1 d", b=num_pixels_y)
    result[:, :, 1, 1:3] = coords
    print(result.shape)
    print(result)
    result = einops.rearrange(result, "y z c d -> (y z) c d")
    return result
    raise NotImplementedError()


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
