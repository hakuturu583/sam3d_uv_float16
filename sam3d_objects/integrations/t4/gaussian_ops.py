"""Apply an affine map to a SAM 3D ``Gaussian`` splat cloud.

The splat attributes are not just positions: the per-splat rotation quaternion
and scale vector describe a covariance ``R diag(s^2) R^T`` that has to be carried
through the same transform, or the reconstruction ends up correctly placed but
smeared along the wrong axes.

Rotation conversions go through ``roma`` (a core dependency, pure python, CPU
safe) rather than being hand-rolled, so there is one fewer quaternion convention
to keep correct here. ``roma`` orders quaternions ``xyzw`` while 3DGS and this
package use ``wxyz``, hence the converters at each boundary.

The tensor-level helper takes plain tensors so it can be exercised on CPU;
``Gaussian`` itself allocates its biases with ``.cuda()`` and needs a GPU.
"""

from __future__ import annotations

import copy

import numpy as np
import roma
import torch

from .frames import matrix_to_quat

__all__ = [
    "kernel_safe_scaling",
    "opaque_positions",
    "transform_gaussian",
    "transform_splats",
]


def _to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    return roma.quat_xyzw_to_wxyz(quaternion)


def _to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    return roma.quat_wxyz_to_xyzw(quaternion)


def _quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose ``(..., 4)`` ``wxyz`` rotations, applying ``b`` first then ``a``."""
    return _to_wxyz(roma.quat_product(_to_xyzw(a), _to_xyzw(b)))


def _similarity_scale(linear: np.ndarray, tol: float = 1e-6) -> float | None:
    """Return ``s`` if ``linear`` is ``s R`` with ``s > 0`` and ``R`` a rotation.

    Returns ``None`` for anything else, including mirroring maps: those have the
    same Gram matrix but no quaternion, so they need the general path.
    """
    gram = linear.T @ linear
    scale_sq = gram[0, 0]
    if scale_sq <= 0.0 or np.linalg.det(linear) <= 0.0:
        return None
    if not np.allclose(gram, scale_sq * np.eye(3), rtol=tol, atol=tol * abs(scale_sq)):
        return None
    return float(np.sqrt(scale_sq))


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

        ``A`` is inspected on the host -- it is a single 3x3, and doing it on the
        device would stall the pipeline three times to read scalars back.
    """
    dtype, device = xyz.dtype, xyz.device
    host_linear = np.asarray(
        linear.detach().cpu().numpy() if torch.is_tensor(linear) else linear, dtype=np.float64
    ).reshape(3, 3)
    lin = torch.as_tensor(host_linear, dtype=dtype, device=device)
    trans = torch.as_tensor(translation, dtype=dtype, device=device).reshape(3)

    new_xyz = xyz @ lin.T + trans

    scale = _similarity_scale(host_linear)
    if scale is not None:
        quat_rot = torch.as_tensor(
            matrix_to_quat(host_linear / scale), dtype=dtype, device=device
        )
        new_quat = _quaternion_multiply(quat_rot.expand_as(quaternion), quaternion)
        return new_xyz, torch.nn.functional.normalize(new_quat, dim=-1), scaling * scale

    # M = A R diag(s), so M M^T is the transformed covariance and the SVD of M
    # reads off its axes directly. `R diag(s)` is a column scaling of R.
    rot_mat = roma.unitquat_to_rotmat(_to_xyzw(quaternion))
    mixed = lin @ (rot_mat * scaling[..., None, :])
    u, singular_values, _ = torch.linalg.svd(mixed)
    # U S^2 U^T is invariant to flipping a column of U, so we are free to make U
    # a proper rotation.
    u[..., 2] *= torch.linalg.det(u).sign()[..., None]
    return new_xyz, _to_wxyz(roma.rotmat_to_unitquat(u)), singular_values


def kernel_safe_scaling(scaling: torch.Tensor, kernel: float, *, margin: float = 1e-6):
    """Lift ``scaling`` clear of the 3D filter floor so ``from_scaling`` stays finite.

    ``Gaussian.from_scaling`` inverts the filter with ``sqrt(s^2 - k^2)``. Every
    transformed radius clears the scaled floor in exact arithmetic (see
    :func:`transform_gaussian`), but the decoder emits most splats *at* the
    floor -- 96% of a decoded cloud has an axis equal to ``k`` -- so in float32
    the difference lands a hair below zero and the square root returns NaN.
    ``save_ply`` then writes those NaNs out, silently deleting most of the cloud.
    """
    if kernel <= 0.0:
        return scaling
    return scaling.clamp_min(kernel * (1.0 + margin))


def opaque_positions(gaussian, threshold: float = 0.9, min_count: int = 32) -> torch.Tensor:
    """Canonical positions of the opaque splats, for measuring the object size.

    Falls back to every splat when the opacity threshold keeps too few, which
    happens on thin or heavily occluded objects.
    """
    xyz = gaussian.get_xyz
    mask = (gaussian.get_opacity > threshold).squeeze(-1)
    return xyz[mask] if int(mask.sum()) >= min_count else xyz


def transform_gaussian(gaussian, linear, translation):
    """Apply ``p -> linear @ p + translation`` to a copy of a :class:`Gaussian`.

    Args:
        gaussian: A ``sam3d_objects...representations.gaussian.Gaussian``.
        linear: ``(3, 3)`` linear map, e.g. ``BoxAlignment.linear``.
        translation: ``(3,)`` translation, e.g. ``BoxAlignment.translation``.

    Returns:
        The transformed ``Gaussian``.
    """
    # A shallow copy is enough: `Gaussian` never mutates a tensor in place -- the
    # `from_*` setters rebind -- so the untouched opacity and colour tensors can
    # be shared with the original instead of duplicated on the device.
    gs = copy.copy(gaussian)

    new_xyz, new_quat, new_scaling = transform_splats(
        gs.get_xyz,
        gs.get_rotation,
        gs.get_scaling,
        linear,
        translation,
    )

    # The 3D filter kernel is a floor on the splat size, so it scales with the
    # object. The smallest singular value keeps the floor below every transformed
    # splat radius, which is what stops `from_scaling`'s sqrt(s^2 - k^2) going NaN.
    kernel_scale = float(np.linalg.svd(np.asarray(linear, dtype=np.float64), compute_uv=False).min())
    new_kernel = gs.mininum_kernel_size * kernel_scale
    gs.mininum_kernel_size = new_kernel

    gs.from_xyz(new_xyz)
    gs.from_rotation(new_quat)
    # The inequality above holds in exact arithmetic but is *tight*: the splats
    # sitting on the floor need `kernel_safe_scaling` to survive float32.
    gs.from_scaling(kernel_safe_scaling(new_scaling, new_kernel))
    return gs
