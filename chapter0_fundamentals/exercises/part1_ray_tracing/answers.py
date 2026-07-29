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
    L = L1 - L2
    L = einops.repeat(L, "h w -> (nrays h) w", nrays=rays.shape[0])
    D = einops.repeat(D, "h w -> (h nsegments) w", nsegments=segments.shape[0])
    A = t.stack([D, L], dim=-1)
    L1 = einops.repeat(L1, "h w -> (nrays h) w", nrays=rays.shape[0])
    O = einops.repeat(O, "h w -> (h nsegments) w", nsegments=segments.shape[0])
    B = L1 - O
    determinants = t.linalg.det(A)
    determinants = determinants.abs() < 1e-8
    A[determinants] = t.eye(2)
    solution = t.linalg.solve(A, B)
    solution = (
        (solution[:, 0] >= 0)
        & (solution[:, 1] <= 1)
        & (0 <= solution[:, 1])
        & ~determinants
    )
    solution = einops.rearrange(solution, "(b1 b) -> b1 b", b=segments.shape[0])
    solution = solution.any(dim=-1)
    return solution
    raise NotImplementedError()


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


# %%
def make_rays_2d(
    num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float
) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    result = t.zeros((num_pixels_y, num_pixels_z, 2, 3))
    result[:, :, 1, 0] = 1
    y_coords = t.linspace(-y_limit, y_limit, num_pixels_y)
    z_coords = t.linspace(-z_limit, z_limit, num_pixels_z)
    coords = t.cartesian_prod(y_coords, z_coords)
    coords = einops.rearrange(coords, "(b1 b) d-> b b1 d", b=num_pixels_y)
    result[:, :, 1, 1:3] = coords
    print(result.shape)
    print(result)
    result = einops.rearrange(result, "y z c d -> (y z) c d")
    return result
    raise NotImplementedError()


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
# %%
# triangle_ray_intersects
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    matrix = t.stack([-D, B - A, C - A], dim=-1)
    vec = O - A
    s, u, v = t.linalg.solve(matrix, vec)
    intersect = (u + v) <= 1 and u >= 0 and v >= 0
    intersect = intersect and s >= 0
    return intersect
    raise NotImplementedError()


tests.test_triangle_ray_intersects(triangle_ray_intersects)


# %%
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    nrays = rays.shape[0]
    A = einops.repeat(triangle[0], "w -> nrays w ", nrays=nrays)
    B = einops.repeat(triangle[1], "w -> nrays w ", nrays=nrays)
    C = einops.repeat(triangle[2], "w -> nrays w ", nrays=nrays)
    O = rays[:, 0, :]
    D = rays[:, 1, :]
    matrix = t.stack([-D, B - A, C - A], dim=-1)
    assert matrix.shape[0] == nrays
    vec = O - A
    assert matrix.shape[0] == vec.shape[0]
    assert matrix.shape[1] == matrix.shape[2]
    assert vec.shape[1] == matrix.shape[1]
    # Missing non singular verification
    determinants = t.linalg.det((matrix))
    singulars = determinants.abs() < 1e-8
    matrix[singulars] = t.eye(3)
    solution = t.linalg.solve(matrix, vec)
    intersect = (solution >= 0).all(dim=-1)
    intersect = intersect & (solution[:, 1] + solution[:, 2] <= 1) & ~singulars
    return intersect
    raise NotImplementedError()


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")


# %%
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    NR = rays.shape[0]

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)
    O, D = rays.unbind(1)

    mat = t.stack([-D, B - A, C - A], dim=-1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=1)

    return (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")


# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    nrays = rays.shape[0]
    nt = triangles.shape[0]
    print(f"nrays={nrays} , nt={nt}")
    A = einops.repeat(triangles[:, 0], "nt w -> nt nrays w ", nrays=nrays)
    B = einops.repeat(triangles[:, 1], "nt w -> nt nrays w ", nrays=nrays)
    C = einops.repeat(triangles[:, 2], "nt w -> nt nrays w ", nrays=nrays)
    assert A.shape[0] == nt
    assert B.shape[0] == nt
    assert C.shape[0] == nt
    assert A.shape[1] == nrays
    assert B.shape[1] == nrays
    assert C.shape[1] == nrays

    O = einops.repeat(rays[:, 0, :], " nrays w -> nt nrays w", nt=nt)
    D = einops.repeat(rays[:, 1, :], " nrays w -> nt nrays w", nt=nt)
    matrix = t.stack([-D, B - A, C - A], dim=-1)
    assert matrix.shape[0] == nt
    assert matrix.shape[1] == nrays

    vec = O - A
    assert matrix.shape[0] == vec.shape[0]
    assert matrix.shape[2] == matrix.shape[3]
    assert vec.shape[1] == matrix.shape[1]

    determinants = t.linalg.det((matrix))
    singulars = determinants.abs() < 1e-8
    matrix[singulars] = t.eye(3)
    solution = t.linalg.solve(matrix, vec)
    print(solution.shape)
    s, u, v = solution.unbind(dim=-1)
    intersect = (s >= 0) & (u >= 0) & (v >= 0) & (u + v <= 1)
    distances = t.zeros(intersect.shape)
    distances[~intersect] = float('inf')
    distances[intersect] = s[intersect]
    distances = distances.T.min(dim=1).values
    print(distances.shape)
    # intersect = intersect & (solution[:,1]+solution[:,2]<=1) & ~singulars
    return intersect.T
    raise NotImplementedError()


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(
    img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000
)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()
