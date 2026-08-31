"""Densification tests: fill the holes between splats, invent nothing else.

Run on CPU with plain tensors; `Gaussian` itself allocates its biases on CUDA.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

torch = pytest.importorskip("torch")
pytest.importorskip("roma")
pytest.importorskip("scipy")

from sam3d_objects.integrations.t4.densify import densify_splats


def lattice(spacing: float = 1.0, side: int = 5, sigma: float = 0.1):
    """A flat grid of identical splats, ``spacing`` apart and too small to touch."""
    axis = torch.arange(side, dtype=torch.float32) * spacing
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
    xyz = torch.cat([grid, torch.zeros(len(grid), 1)], dim=-1)
    n = len(xyz)
    quat = torch.zeros(n, 4)
    quat[:, 0] = 1.0
    scaling = torch.full((n, 3), sigma)
    opacity = torch.full((n, 1), 0.5)
    features = torch.zeros(n, 3, 1)
    return xyz, quat, scaling, opacity, features


def nearest_spacing(xyz):
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(xyz.numpy()).query(xyz.numpy(), k=2)
    return float(np.median(distance[:, 1]))


def test_holes_are_filled_and_spacing_halves():
    splats = lattice()
    before = nearest_spacing(splats[0])
    xyz, quat, scaling, opacity, features = densify_splats(*splats, passes=1)

    assert len(xyz) > len(splats[0])
    assert nearest_spacing(xyz) == pytest.approx(before / 2, rel=1e-3)
    # Every array grows together, or the Gaussian would be inconsistent.
    assert len(quat) == len(scaling) == len(opacity) == len(features) == len(xyz)
    assert torch.allclose(torch.linalg.norm(quat, dim=-1), torch.ones(len(quat)), atol=1e-5)
    assert torch.isfinite(xyz).all() and torch.isfinite(scaling).all()


def test_a_second_pass_keeps_closing_the_remaining_holes():
    splats = lattice()
    one = densify_splats(*splats, passes=1)[0]
    two = densify_splats(*splats, passes=2)[0]
    assert len(two) > len(one)
    assert nearest_spacing(two) < nearest_spacing(one)


def test_overlapping_splats_are_left_alone():
    """Nothing to fill when the radii already span the spacing."""
    splats = lattice(sigma=1.0)
    xyz = densify_splats(*splats, passes=3)[0]
    assert len(xyz) == len(splats[0])


def test_real_empty_space_is_not_bridged():
    """A gap far wider than the lattice is geometry, not a sampling hole."""
    left = lattice(side=4)
    right = list(lattice(side=4))
    right[0] = right[0] + torch.tensor([20.0, 0.0, 0.0])
    splats = [torch.cat([a, b]) for a, b in zip(left, right)]

    xyz = densify_splats(*splats, passes=1)[0]
    middle = (xyz[:, 0] > 4.5) & (xyz[:, 0] < 19.5)
    assert not middle.any(), "densification jumped the gap between the two clusters"


def test_interpolated_splats_average_their_parents():
    xyz, quat, scaling, opacity, features = lattice(side=2)
    opacity = torch.tensor([[0.2], [0.4], [0.6], [0.8]])
    features = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1).expand(4, 3, 1).contiguous()

    out = densify_splats(xyz, quat, scaling, opacity, features, passes=1)
    new_opacity, new_features = out[3][4:], out[4][4:]

    assert new_opacity.min() >= 0.2 and new_opacity.max() <= 0.8
    assert new_features.min() >= 0.0 and new_features.max() <= 3.0
    # Identical parent radii: the geometric mean is that radius exactly.
    assert torch.allclose(out[2][4:], torch.full_like(out[2][4:], 0.1), atol=1e-6)
