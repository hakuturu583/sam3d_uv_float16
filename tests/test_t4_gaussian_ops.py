"""Splat-level tests: a transform must carry the Gaussian covariance with it.

Run on CPU with plain tensors; `Gaussian` itself allocates its biases on CUDA.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

torch = pytest.importorskip("torch")

from sam3d_objects.integrations.t4.frames import VIEWER_AXES, quat_to_matrix, rotz
from sam3d_objects.integrations.t4.gaussian_ops import (
    _matrix_to_quaternion,
    _quaternion_to_matrix,
    quaternion_multiply,
    transform_splats,
)


def random_splats(n: int = 256, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    xyz = torch.randn(n, 3, generator=generator)
    quat = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator), dim=-1)
    scaling = torch.rand(n, 3, generator=generator) * 0.1 + 0.01
    return xyz, quat, scaling


def covariance(quat, scaling):
    rot = _quaternion_to_matrix(quat)
    return rot @ torch.diag_embed(scaling**2) @ rot.transpose(-1, -2)


def test_quaternion_matrix_round_trip_matches_numpy():
    _, quat, _ = random_splats(64)
    matrices = _quaternion_to_matrix(quat)
    for i in range(0, 64, 7):
        assert np.allclose(matrices[i].numpy(), quat_to_matrix(quat[i].numpy()), atol=1e-6)
    recovered = _matrix_to_quaternion(matrices)
    # q and -q are the same rotation
    assert torch.allclose(_quaternion_to_matrix(recovered), matrices, atol=1e-5)


def test_quaternion_multiply_composes_rotations():
    a = torch.nn.functional.normalize(torch.tensor([[0.3, 0.1, -0.4, 0.8]]), dim=-1)
    b = torch.nn.functional.normalize(torch.tensor([[-0.2, 0.7, 0.1, 0.5]]), dim=-1)
    composed = _quaternion_to_matrix(quaternion_multiply(a, b))
    assert torch.allclose(composed, _quaternion_to_matrix(a) @ _quaternion_to_matrix(b), atol=1e-6)


@pytest.mark.parametrize(
    "linear",
    [
        torch.eye(3) * 3.7,
        torch.tensor(rotz(0.9), dtype=torch.float32) * 2.0,
        torch.tensor(VIEWER_AXES["gltf"], dtype=torch.float32),
        torch.diag(torch.tensor([1.4, 0.6, 2.2])),
        torch.tensor([[0.9, 0.3, 0.0], [-0.2, 1.1, 0.4], [0.0, 0.1, 0.7]]),
    ],
)
def test_transform_preserves_the_gaussian_covariance(linear):
    """`Sigma -> A Sigma A^T` must hold for similarities and general maps alike."""
    xyz, quat, scaling = random_splats()
    translation = torch.tensor([1.0, -2.0, 0.5])

    new_xyz, new_quat, new_scaling = transform_splats(xyz, quat, scaling, linear, translation)

    assert torch.allclose(new_xyz, xyz @ linear.T + translation, atol=1e-5)
    expected = linear @ covariance(quat, scaling) @ linear.T
    assert torch.allclose(covariance(new_quat, new_scaling), expected, atol=1e-5)
    assert torch.all(new_scaling > 0)


def test_similarity_scales_the_splat_radii_uniformly():
    xyz, quat, scaling = random_splats()
    linear = torch.tensor(rotz(0.4), dtype=torch.float32) * 2.5
    _, _, new_scaling = transform_splats(xyz, quat, scaling, linear, torch.zeros(3))
    assert torch.allclose(new_scaling, scaling * 2.5, atol=1e-5)


def test_mirroring_map_is_handled_by_the_general_path():
    """A negative determinant has no quaternion; the covariance must still be right."""
    xyz, quat, scaling = random_splats(64)
    linear = torch.diag(torch.tensor([1.0, -1.0, 1.0])) * 2.0
    _, new_quat, new_scaling = transform_splats(xyz, quat, scaling, linear, torch.zeros(3))
    expected = linear @ covariance(quat, scaling) @ linear.T
    assert torch.allclose(covariance(new_quat, new_scaling), expected, atol=1e-5)
    assert torch.allclose(torch.linalg.norm(new_quat, dim=-1), torch.ones(64), atol=1e-5)


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
