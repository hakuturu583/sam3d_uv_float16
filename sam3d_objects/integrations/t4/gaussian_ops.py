"""Apply an affine map to a SAM 3D ``Gaussian`` splat cloud.

The splat attributes are not just positions: the per-splat rotation quaternion
and scale vector describe a covariance ``R diag(s^2) R^T`` that has to be carried
through the same transform, or the reconstruction ends up correctly placed but
smeared along the wrong axes.

The tensor-level helpers take plain tensors so they can be exercised on CPU;
``Gaussian`` itself allocates its biases with ``.cuda()`` and needs a GPU.
"""

from __future__ import annotations

from copy import deepcopy

import torch

__all__ = [
    "opaque_mask",
    "opaque_positions",
    "quaternion_multiply",
    "transform_gaussian",
    "transform_splats",
]


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of ``(..., 4)`` quaternions in ``(w, x, y, z)`` order."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` rotation matrices to ``(..., 4)`` ``(w, x, y, z)`` quaternions."""
    m = matrix
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    # Build all four candidate quaternions and pick the numerically best one,
    # which keeps this branch-free and safe to run over a million splats.
    q_abs = torch.stack(
        (
            1.0 + trace,
            1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        ),
        dim=-1,
    ).clamp_min(0.0).sqrt()

    candidates = torch.stack(
        (
            torch.stack(
                (
                    q_abs[..., 0] ** 2,
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m[..., 2, 1] - m[..., 1, 2],
                    q_abs[..., 1] ** 2,
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    q_abs[..., 2] ** 2,
                    m[..., 2, 1] + m[..., 1, 2],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 2, 1] + m[..., 1, 2],
                    q_abs[..., 3] ** 2,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )

    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat = candidates / (2.0 * q_abs[..., None].max(flr))
    best = q_abs.argmax(dim=-1)
    quat = quat.gather(-2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    return torch.nn.functional.normalize(quat, dim=-1)


def _is_similarity(linear: torch.Tensor, tol: float = 1e-6) -> bool:
    """True for ``A = s R`` with ``s > 0`` and ``R`` a proper rotation.

    A mirroring map has the same Gram matrix but no quaternion, so it is
    excluded here and handled by the general SVD path.
    """
    if float(torch.linalg.det(linear)) <= 0.0:
        return False
    gram = linear.T @ linear
    scale_sq = gram[0, 0]
    eye = torch.eye(3, dtype=linear.dtype, device=linear.device)
    return bool(torch.allclose(gram, scale_sq * eye, rtol=tol, atol=tol * float(scale_sq.abs())))


def transform_splats(xyz, quaternion, scaling, linear, translation):
    """Push splat positions/orientations/scales through ``p -> A p + t``.

    Args:
        xyz: ``(N, 3)`` splat centres.
        quaternion: ``(N, 4)`` splat rotations, ``(w, x, y, z)``, column-vector
            convention (matches 3DGS ``build_rotation`` and PyTorch3D).
        scaling: ``(N, 3)`` splat scales (activated, i.e. ``Gaussian.get_scaling``).
        linear: ``(3, 3)`` linear map ``A``.
        translation: ``(3,)`` translation ``t``.

    Returns:
        ``(xyz, quaternion, scaling)`` after the transform.

    Notes:
        For a similarity ``A = s R`` the update is exact and cheap: rotate the
        quaternion, multiply the scales by ``s``. For a general ``A`` the new
        covariance is ``(A R S)(A R S)^T``; its SVD gives the new rotation and
        scales exactly, at the cost of a batched 3x3 SVD.
    """
    dtype = xyz.dtype
    device = xyz.device
    lin = torch.as_tensor(linear, dtype=dtype, device=device).reshape(3, 3)
    trans = torch.as_tensor(translation, dtype=dtype, device=device).reshape(3)

    new_xyz = xyz @ lin.T + trans

    if _is_similarity(lin.double()):
        scale = torch.linalg.det(lin.double()).abs().pow(1.0 / 3.0).to(dtype)
        rot = lin / scale
        quat_rot = _matrix_to_quaternion(rot[None].double())[0].to(dtype)
        new_quat = quaternion_multiply(quat_rot.expand_as(quaternion), quaternion)
        new_scaling = scaling * scale
        return new_xyz, torch.nn.functional.normalize(new_quat, dim=-1), new_scaling

    rot_mat = _quaternion_to_matrix(quaternion)
    mixed = lin[None] @ rot_mat @ torch.diag_embed(scaling)
    u, sing, _ = torch.linalg.svd(mixed.double())
    # A negative determinant flips a column of U; the covariance U S^2 U^T is
    # unchanged by that flip, so we are free to make U a proper rotation.
    sign = torch.linalg.det(u).sign()
    u = torch.cat((u[..., :2], u[..., 2:] * sign[..., None, None]), dim=-1)
    new_quat = _matrix_to_quaternion(u).to(dtype)
    return new_xyz, new_quat, sing.to(dtype)


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """``(..., 4)`` ``(w, x, y, z)`` quaternions to ``(..., 3, 3)`` rotation matrices."""
    q = torch.nn.functional.normalize(quaternion, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def opaque_mask(gaussian, threshold: float = 0.9) -> torch.Tensor:
    """Boolean mask of splats whose opacity exceeds ``threshold``."""
    return (gaussian.get_opacity > threshold).squeeze(-1)


def opaque_positions(gaussian, threshold: float = 0.9, min_count: int = 32) -> torch.Tensor:
    """Canonical positions of the opaque splats, for measuring the object size.

    Falls back to every splat when the opacity threshold keeps too few, which
    happens on thin or heavily occluded objects.
    """
    xyz = gaussian.get_xyz
    mask = opaque_mask(gaussian, threshold)
    return xyz[mask] if int(mask.sum()) >= min_count else xyz


def transform_gaussian(gaussian, linear, translation, in_place: bool = False):
    """Apply ``p -> linear @ p + translation`` to a :class:`Gaussian` in place.

    Args:
        gaussian: A ``sam3d_objects...representations.gaussian.Gaussian``.
        linear: ``(3, 3)`` linear map, e.g. ``BoxAlignment.linear``.
        translation: ``(3,)`` translation, e.g. ``BoxAlignment.translation``.
        in_place: Mutate ``gaussian`` instead of a deep copy.

    Returns:
        The transformed ``Gaussian``.
    """
    gs = gaussian if in_place else deepcopy(gaussian)

    new_xyz, new_quat, new_scaling = transform_splats(
        gs.get_xyz,
        gs.get_rotation,
        gs.get_scaling,
        linear,
        translation,
    )

    lin = torch.as_tensor(linear, dtype=torch.float64)
    # The 3D filter kernel is a floor on the splat size, so it scales with the
    # object. Use the smallest singular value so the floor never grows past what
    # the transform actually did to the tightest axis.
    kernel_scale = float(torch.linalg.svdvals(lin).min())
    gs.mininum_kernel_size = gs.mininum_kernel_size * kernel_scale

    gs.from_xyz(new_xyz)
    gs.from_rotation(new_quat)
    gs.from_scaling(new_scaling)
    return gs
