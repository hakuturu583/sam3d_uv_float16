"""Splat-level tests: a transform must carry the Gaussian covariance with it.

Run on CPU with plain tensors; `Gaussian` itself allocates its biases on CUDA.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

torch = pytest.importorskip("torch")
roma = pytest.importorskip("roma")

from sam3d_objects.integrations.t4.frames import VIEWER_AXES, quat_to_matrix, rotz
from sam3d_objects.integrations.t4.gaussian_ops import (
    _quaternion_multiply,
    _similarity_scale,
    kernel_safe_scaling,
    transform_splats,
)


def random_splats(n: int = 256, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    xyz = torch.randn(n, 3, generator=generator)
    quat = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator), dim=-1)
    scaling = torch.rand(n, 3, generator=generator) * 0.1 + 0.01
    return xyz, quat, scaling


def rotation_of(quat_wxyz):
    """`wxyz` quaternions to rotation matrices, via roma's `xyzw` API."""
    return roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(quat_wxyz))


def covariance(quat, scaling):
    rot = rotation_of(quat)
    return rot @ torch.diag_embed(scaling**2) @ rot.transpose(-1, -2)


def test_numpy_quaternion_convention_matches_roma():
    """`frames.quat_to_matrix` claims wxyz + column vectors; hold it to that."""
    _, quat, _ = random_splats(64)
    reference = rotation_of(quat)
    for i in range(0, 64, 7):
        assert quat_to_matrix(quat[i].numpy()) == pytest.approx(reference[i].numpy(), abs=1e-6)


def test_quaternion_multiply_composes_rotations_in_wxyz_order():
    """Pins the wxyz <-> xyzw boundary: a flipped converter would show up here."""
    a = torch.nn.functional.normalize(torch.tensor([[0.3, 0.1, -0.4, 0.8]]), dim=-1)
    b = torch.nn.functional.normalize(torch.tensor([[-0.2, 0.7, 0.1, 0.5]]), dim=-1)
    composed = rotation_of(_quaternion_multiply(a, b))
    assert torch.allclose(composed, rotation_of(a) @ rotation_of(b), atol=1e-6)


@pytest.mark.parametrize(
    ("linear", "expected_scale"),
    [
        (torch.eye(3) * 3.7, 3.7),
        (torch.tensor(rotz(0.9), dtype=torch.float32) * 2.0, 2.0),
        (torch.tensor(VIEWER_AXES["gltf"], dtype=torch.float32), 1.0),
        (torch.diag(torch.tensor([1.4, 0.6, 2.2])), None),
        (torch.tensor([[0.9, 0.3, 0.0], [-0.2, 1.1, 0.4], [0.0, 0.1, 0.7]]), None),
        (torch.diag(torch.tensor([1.0, -1.0, 1.0])) * 2.0, None),  # mirror: no quaternion
    ],
)
def test_transform_preserves_the_gaussian_covariance(linear, expected_scale):
    """`Sigma -> A Sigma A^T` must hold on both the similarity and the SVD path."""
    xyz, quat, scaling = random_splats()
    translation = torch.tensor([1.0, -2.0, 0.5])

    scale = _similarity_scale(np.asarray(linear, dtype=np.float64))
    if expected_scale is None:
        assert scale is None
    else:
        assert scale == pytest.approx(expected_scale, rel=1e-6)

    new_xyz, new_quat, new_scaling = transform_splats(xyz, quat, scaling, linear, translation)

    assert torch.allclose(new_xyz, xyz @ linear.T + translation, atol=1e-5)
    expected = linear @ covariance(quat, scaling) @ linear.T
    assert torch.allclose(covariance(new_quat, new_scaling), expected, atol=1e-5)
    assert torch.all(new_scaling > 0)
    assert torch.allclose(torch.linalg.norm(new_quat, dim=-1), torch.ones(len(new_quat)), atol=1e-5)


def test_similarity_scales_the_splat_radii_uniformly():
    xyz, quat, scaling = random_splats()
    linear = torch.tensor(rotz(0.4), dtype=torch.float32) * 2.5
    _, _, new_scaling = transform_splats(xyz, quat, scaling, linear, torch.zeros(3))
    assert torch.allclose(new_scaling, scaling * 2.5, atol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_transformed_scales_stay_above_the_scaled_kernel_floor(seed):
    """`Gaussian.from_scaling` takes `sqrt(s^2 - k^2)`, so `s < k` would give NaN.

    `transform_gaussian` scales the 3D filter kernel by the transform's smallest
    singular value; this pins the inequality that makes that safe.
    """
    _, quat, scaling = random_splats(128, seed=seed)
    kernel = float(scaling.min()) * 0.5
    scaling = torch.sqrt(scaling**2 + kernel**2)  # what `get_scaling` returns
    linear = torch.tensor([[1.7, 0.4, 0.0], [-0.3, 2.2, 0.1], [0.0, 0.2, 0.9]])

    _, _, new_scaling = transform_splats(torch.zeros(128, 3), quat, scaling, linear, torch.zeros(3))
    new_kernel = float(torch.linalg.svdvals(linear).min()) * kernel
    assert float(new_scaling.min()) >= new_kernel - 1e-6


def softplus_inverse(x):
    """`gaussian_model.softplus_inverse_scaling_activation`, inlined to stay CPU-only.

    Undefined at zero (`log(0)`) and NaN below it -- which is the whole point here.
    """
    return x + torch.log(-torch.expm1(-x))


def test_floor_splats_survive_the_transform_in_float32():
    """The decoder puts most splats *on* the 3D filter floor; they must survive.

    `get_scaling` reads `sqrt(softplus(...)^2 + k^2)`, and for a floor splat the
    softplus term is far below float32 resolution next to `k`, so the radius comes
    back as exactly `k`. `from_scaling` then inverts with `sqrt(s^2 - k^2)` on a
    radicand that has rounded to zero (or below), and `softplus_inverse` turns
    that into -inf/NaN -- the failure that emptied 93% of an exported T4 cloud.
    """
    kernel = 0.0009  # 3d_filter_kernel_size, from the released decoder config
    n = 512
    _, quat, _ = random_splats(n)
    scaling = torch.full((n, 3), kernel)  # what `get_scaling` returns on the floor

    # The real box refit: a similarity that mirrors X and Y (camera -> box frame).
    linear = torch.tensor(np.diag([-1.0, -1.0, 1.0]) * 9.476282, dtype=torch.float32)
    _, _, new_scaling = transform_splats(torch.zeros(n, 3), quat, scaling, linear, torch.zeros(3))
    # `transform_gaussian` rescales the kernel in float64, as numpy's SVD returns it.
    new_kernel = float(np.linalg.svd(linear.numpy().astype(np.float64), compute_uv=False).min()) * kernel

    from_scaling = lambda s: softplus_inverse(torch.sqrt(torch.square(s) - new_kernel**2))
    assert not torch.isfinite(from_scaling(new_scaling)).all(), (
        "test no longer reproduces the float32 collapse at the kernel floor"
    )

    guarded = from_scaling(kernel_safe_scaling(new_scaling, new_kernel))
    assert torch.isfinite(guarded).all()
    # The guard must not inflate the splats: the filtered radius still reads back
    # as the scaled floor.
    radius = torch.sqrt(torch.nn.functional.softplus(guarded) ** 2 + new_kernel**2)
    assert torch.allclose(radius, new_scaling, rtol=1e-4)


def test_kernel_safe_scaling_leaves_ordinary_splats_alone():
    _, _, scaling = random_splats(64)
    assert torch.equal(kernel_safe_scaling(scaling, 0.0), scaling)
    assert torch.equal(kernel_safe_scaling(scaling, float(scaling.min()) * 0.5), scaling)
