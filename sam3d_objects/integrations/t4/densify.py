"""Fill the lattice gaps in a splat cloud by interpolating new Gaussians.

SAM 3D decodes into a 64^3 voxel grid, so its splats sit on a lattice whose
spacing is a fixed fraction of the canonical cube. Refitting an object to its
true metric size scales positions and radii by the same factor, which leaves the
*ratio* between them -- and therefore the holes -- exactly as they were: a 9 m
truck is as stippled as the 0.9 m canonical one, only ten times wider on screen.
Closing those holes needs more Gaussians, not bigger ones. Inflating the radii
instead blurs the whole surface to hide gaps that only exist between samples.

So this inserts one interpolated splat at the midpoint of every neighbour pair
the lattice left uncovered, and repeats until the pairs are covered. Two rules
keep it from inventing geometry:

* a pair is only filled when its centres are farther apart than the radii they
  span (``coverage``), i.e. when there is a visible hole between them;
* a pair is skipped when it is farther apart than ``max_gap`` times the cloud's
  median neighbour spacing, so genuine empty space -- across a wheel arch, or
  between the cab and the body -- is never bridged.
"""

from __future__ import annotations

import copy

import numpy as np
import roma
import torch

from .gaussian_ops import _to_wxyz, _to_xyzw, kernel_safe_scaling

__all__ = ["densify_gaussian", "densify_splats"]

#: Ceiling on the splat count, so a pathological cloud cannot exhaust device memory.
MAX_SPLATS = 8_000_000


def _uncovered_pairs(xyz, scaling, *, neighbours, coverage, max_gap):
    """Neighbour pairs with a visible hole between them, as ``(2, M)`` indices.

    The k-nearest-neighbour graph is built on the host: ``scipy``'s KD-tree beats
    a dense distance matrix by orders of magnitude at these counts, and the
    positions are a few megabytes.
    """
    from scipy.spatial import cKDTree

    host = xyz.detach().cpu().numpy().astype(np.float64)
    k = min(neighbours + 1, len(host))
    if k < 2:
        return None, 0.0
    distance, index = cKDTree(host).query(host, k=k, workers=-1)

    # Column 0 is the point itself; the rest are its neighbours.
    source = np.repeat(np.arange(len(host)), k - 1)
    target = index[:, 1:].reshape(-1)
    length = distance[:, 1:].reshape(-1)
    spacing = float(np.median(distance[:, 1]))

    # (i, j) and (j, i) describe the same midpoint: keep one.
    keep = source < target
    source, target, length = source[keep], target[keep], length[keep]
    if len(source) == 0:
        return None, spacing

    device = xyz.device
    pair = torch.from_numpy(np.stack([source, target])).to(device)
    gap = torch.from_numpy(length).to(device=device, dtype=xyz.dtype)

    radius = scaling.mean(dim=-1)
    span = coverage * (radius[pair[0]] + radius[pair[1]])
    uncovered = (gap > span) & (gap < max_gap * spacing)
    return pair[:, uncovered], spacing


def densify_splats(
    xyz,
    quaternion,
    scaling,
    opacity,
    features,
    *,
    neighbours: int = 6,
    coverage: float = 2.0,
    max_gap: float = 2.5,
    passes: int = 2,
    max_splats: int = MAX_SPLATS,
):
    """Insert interpolated splats into the holes of a splat cloud.

    Args:
        xyz: ``(N, 3)`` splat centres.
        quaternion: ``(N, 4)`` rotations, ``(w, x, y, z)``.
        scaling: ``(N, 3)`` activated scales, i.e. ``Gaussian.get_scaling``.
        opacity: ``(N, 1)`` activated opacities.
        features: ``(N, C, S)`` spherical-harmonic coefficients.
        neighbours: Size of the k-nearest-neighbour graph each pass considers.
        coverage: A pair is a hole when its length exceeds ``coverage`` times the
            sum of the two mean radii. Larger values fill less.
        max_gap: Never bridge a pair longer than this many median spacings.
        passes: How many times to repeat; each pass halves the holes it fills, so
            two passes close a gap of up to four times the splat spacing.
        max_splats: Stop early rather than grow past this count.

    Returns:
        ``(xyz, quaternion, scaling, opacity, features)``, the originals with the
        new splats appended. The inputs are never mutated.
    """
    for _ in range(max(passes, 0)):
        if len(xyz) >= max_splats:
            break
        pair, _ = _uncovered_pairs(
            xyz, scaling, neighbours=neighbours, coverage=coverage, max_gap=max_gap
        )
        if pair is None or pair.shape[1] == 0:
            break
        if len(xyz) + pair.shape[1] > max_splats:
            pair = pair[:, : max_splats - len(xyz)]

        left, right = pair[0], pair[1]
        # Slerp keeps the interpolated orientation a unit quaternion; a plain mean
        # would shrink it toward zero for widely separated rotations.
        mid_quat = roma.utils.unitquat_slerp(
            _to_xyzw(quaternion[left]), _to_xyzw(quaternion[right]), torch.tensor([0.5])
        )[0]

        xyz = torch.cat([xyz, (xyz[left] + xyz[right]) * 0.5])
        quaternion = torch.cat([quaternion, _to_wxyz(mid_quat)])
        # Geometric mean on the radii: scales live on a log axis, and it keeps the
        # new splat from bulging past the smaller of its two parents.
        scaling = torch.cat([scaling, torch.sqrt(scaling[left] * scaling[right])])
        opacity = torch.cat([opacity, (opacity[left] + opacity[right]) * 0.5])
        features = torch.cat([features, (features[left] + features[right]) * 0.5])

    return xyz, quaternion, scaling, opacity, features


def densify_gaussian(gaussian, **kwargs):
    """Return a copy of ``gaussian`` with its lattice holes filled.

    Keyword arguments are forwarded to :func:`densify_splats`.
    """
    gs = copy.copy(gaussian)
    xyz, quaternion, scaling, opacity, features = densify_splats(
        gs.get_xyz,
        gs.get_rotation,
        gs.get_scaling,
        gs.get_opacity,
        gs._features_dc,
        **kwargs,
    )
    gs.from_xyz(xyz)
    gs.from_rotation(quaternion)
    gs.from_scaling(kernel_safe_scaling(scaling, gs.mininum_kernel_size))
    gs.from_opacity(opacity)
    gs.from_features(features)
    return gs
