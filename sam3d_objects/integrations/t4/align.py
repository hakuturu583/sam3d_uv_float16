"""Fit a SAM 3D Objects reconstruction onto a T4 3D bounding box.

SAM 3D gives a shape in a unit cube plus a layout ``(rotation, translation,
scale)`` whose metric scale comes from MoGe and is therefore only defined up to
the point map's own scale/shift normalisation. A T4 annotation gives the exact
metric size and heading. This module combines the two:

* the **rotation** is snapped so the reconstruction is axis-aligned with the box,
  which makes a car point along the box's +X (forward) axis;
* the **scale** is refitted from the box's ``(width, length, height)``;
* the **translation** is re-derived from the box centre, since SAM 3D's is in
  MoGe units.

Everything here is ``numpy`` only -- no torch, no CUDA, no ``t4-devkit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .frames import (
    matrix_to_quat,
    obj_to_box_pose,
    rotz,
    snap_to_axis_rotation,
    snap_yaw_only,
    yaw_of,
)

__all__ = ["BoxAlignment", "align_to_box", "compose_alignment", "robust_extent"]

ROTATION_MODES = ("snap24", "yaw", "none")
SCALE_MODES = ("iso", "length", "width", "height", "axis", "none")
Z_ALIGN_MODES = ("center", "bottom")


@dataclass(frozen=True)
class BoxAlignment:
    """An affine map taking SAM 3D canonical object points into a target frame.

    ``p_out = linear @ p_obj + translation``, where ``p_obj`` is
    ``Gaussian.get_xyz`` (the ``[-0.5, 0.5]^3`` cube).

    Attributes:
        linear: ``(3, 3)`` linear part. A similarity (``s * R``) unless
            ``scale_mode="axis"`` was used, in which case it is a general
            invertible matrix.
        translation: ``(3,)`` translation.
        frame: Name of the frame ``linear``/``translation`` map into.
        report: Diagnostics -- see :func:`align_to_box`.
    """

    linear: np.ndarray
    translation: np.ndarray
    frame: str
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def is_similarity(self) -> bool:
        """True when ``linear`` is a uniform scale times a rotation."""
        gram = self.linear.T @ self.linear
        scale_sq = float(gram[0, 0])
        return bool(np.allclose(gram, scale_sq * np.eye(3), rtol=1e-6, atol=1e-9))

    @property
    def scale(self) -> float:
        """Uniform scale factor. Raises for anisotropic alignments."""
        if not self.is_similarity:
            raise ValueError("alignment is not a similarity; inspect `linear` instead")
        return float(np.sqrt(np.linalg.det(self.linear) ** (2.0 / 3.0)))

    @property
    def rotation(self) -> np.ndarray:
        """Rotation part. Raises for anisotropic alignments."""
        return self.linear / self.scale

    @property
    def quaternion(self) -> np.ndarray:
        """Rotation part as a ``(w, x, y, z)`` quaternion."""
        return matrix_to_quat(self.rotation)


def robust_extent(points, percentile: float = 1.0):
    """Return ``(lower, upper, extent)`` per axis, ignoring outlier splats.

    3D Gaussian reconstructions almost always carry a few stray low-opacity
    blobs; a raw min/max would let one of them set the metric scale.

    Args:
        points: ``(N, 3)`` array.
        percentile: Percentage trimmed from each end. ``0`` gives a plain
            axis-aligned bounding box.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("no points to measure")
    if percentile <= 0.0:
        lower, upper = pts.min(axis=0), pts.max(axis=0)
    else:
        lower = np.percentile(pts, percentile, axis=0)
        upper = np.percentile(pts, 100.0 - percentile, axis=0)
    return lower, upper, np.maximum(upper - lower, 1e-9)


def align_to_box(
    obj_points,
    *,
    sam3d_rotation,
    sam3d_translation,
    sam3d_scale,
    box_rotation,
    box_position,
    box_size,
    rotation_mode: str = "snap24",
    scale_mode: str = "iso",
    z_align: str = "center",
    keep_translation: bool = False,
    extra_yaw_deg: float = 0.0,
    percentile: float = 1.0,
) -> BoxAlignment:
    """Solve for the map that drops a SAM 3D reconstruction into its T4 box.

    Args:
        obj_points: ``(N, 3)`` canonical Gaussian centres (``Gaussian.get_xyz``).
            Pre-filter these to the opaque splats -- see
            :func:`~sam3d_objects.integrations.t4.gaussian_ops.opaque_positions`.
        sam3d_rotation: ``output["rotation"]`` quaternion ``(w, x, y, z)``.
        sam3d_translation: ``output["translation"]``.
        sam3d_scale: ``output["scale"]``.
        box_rotation: ``(3, 3)`` box-to-camera rotation, from a ``Box3D``
            retrieved with ``as_sensor_coord=True`` for the camera the image
            came from.
        box_position: ``(3,)`` box centre in that same camera frame.
        box_size: ``(width, length, height)`` -- T4's ``Box3D.size`` order.
        rotation_mode: ``"snap24"`` snaps onto the nearest axis-aligned rotation
            so the object is exactly square with the box (recommended, and what
            makes a car face forward). ``"yaw"`` corrects the heading only and
            preserves any roll/pitch SAM 3D predicted. ``"none"`` keeps SAM 3D's
            rotation untouched.
        scale_mode: which box dimension drives the metric fit. ``"iso"`` takes
            the median of the three per-axis ratios, ``"length"``/``"width"``/
            ``"height"`` use a single axis, ``"axis"`` stretches each axis
            independently (exact box fill, distorts the shape), ``"none"`` keeps
            SAM 3D's MoGe-derived scale.
        z_align: ``"center"`` puts the reconstruction's centre at the box centre;
            ``"bottom"`` seats its underside on the bottom face of the box, which
            usually looks better for vehicles.
        keep_translation: Keep SAM 3D's predicted offset from the box centre
            (converted to metres) instead of re-centring on the box.
        extra_yaw_deg: Manual heading offset applied after the correction. SAM 3D
            occasionally reads a symmetric car back-to-front; ``180`` turns it
            around without touching anything else.
        percentile: Outlier trim used when measuring the reconstruction, in %.

    Returns:
        A :class:`BoxAlignment` into the ``"box"`` frame. Use
        :func:`compose_alignment` to move it to camera / ``base_link`` / map.
    """
    if rotation_mode not in ROTATION_MODES:
        raise ValueError(f"rotation_mode must be one of {ROTATION_MODES}, got {rotation_mode!r}")
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}")
    if z_align not in Z_ALIGN_MODES:
        raise ValueError(f"z_align must be one of {Z_ALIGN_MODES}, got {z_align!r}")

    rot_box_obj, trans_box, scale_sam3d = obj_to_box_pose(
        sam3d_rotation=sam3d_rotation,
        sam3d_translation=sam3d_translation,
        sam3d_scale=sam3d_scale,
        box_rotation=box_rotation,
        box_position=box_position,
    )

    # --- rotation -----------------------------------------------------------
    snapped, snap_angle = snap_to_axis_rotation(rot_box_obj)
    if rotation_mode == "snap24":
        rotation = snapped
        yaw_delta = yaw_of(snapped @ rot_box_obj.T)
    elif rotation_mode == "yaw":
        rotation, yaw_delta = snap_yaw_only(rot_box_obj)
    else:
        rotation, yaw_delta = rot_box_obj, 0.0

    if extra_yaw_deg:
        rotation = rotz(np.radians(extra_yaw_deg)) @ rotation
        yaw_delta += np.radians(extra_yaw_deg)

    # --- scale --------------------------------------------------------------
    # Measure the reconstruction *after* rotating it, so the extents line up with
    # the box axes: X is length, Y is width, Z is height.
    pts = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    rotated = pts @ rotation.T
    lower, upper, extent = robust_extent(rotated, percentile=percentile)

    width, length, height = np.asarray(box_size, dtype=np.float64).reshape(3)
    target = np.array([length, width, height])  # box frame axis order: X, Y, Z
    ratios = target / extent

    if scale_mode == "iso":
        scale = np.full(3, float(np.median(ratios)))
    elif scale_mode == "length":
        scale = np.full(3, float(ratios[0]))
    elif scale_mode == "width":
        scale = np.full(3, float(ratios[1]))
    elif scale_mode == "height":
        scale = np.full(3, float(ratios[2]))
    elif scale_mode == "axis":
        scale = ratios.astype(np.float64)
    else:  # "none"
        scale = np.full(3, scale_sam3d)

    linear = np.diag(scale) @ rotation

    # --- translation --------------------------------------------------------
    center = scale * (lower + upper) / 2.0
    if keep_translation:
        # SAM 3D's offset is in MoGe units; rescale it by the same factor the
        # shape was rescaled by so the two stay consistent.
        offset = trans_box * float(np.mean(scale)) / max(scale_sam3d, 1e-12)
        translation = offset
    else:
        translation = -center
        if z_align == "bottom":
            translation[2] = -scale[2] * lower[2] - height / 2.0

    report = {
        "rotation_mode": rotation_mode,
        "scale_mode": scale_mode,
        "z_align": z_align,
        "sam3d_scale": scale_sam3d,
        "fitted_scale": scale.tolist(),
        "scale_ratio_vs_sam3d": (scale / max(scale_sam3d, 1e-12)).tolist(),
        "snap_angle_deg": float(np.degrees(snap_angle)),
        "yaw_correction_deg": float(np.degrees(yaw_delta)),
        "extra_yaw_deg": float(extra_yaw_deg),
        "measured_extent_obj": extent.tolist(),
        "target_size_xyz": target.tolist(),
        "box_size_wlh": [float(width), float(length), float(height)],
        "sam3d_offset_in_box_m": trans_box.tolist(),
        "axis_map": _describe_axis_map(rotation),
    }
    return BoxAlignment(linear=linear, translation=translation, frame="box", report=report)


def compose_alignment(
    alignment: BoxAlignment,
    rotation,
    translation,
    frame: str,
) -> BoxAlignment:
    """Push an alignment through a further rigid transform.

    ``p_new = rotation @ p_old + translation``.

    Use it to go from the box frame to the camera frame
    (``box.rotation.rotation_matrix``, ``box.position``), then on to
    ``base_link`` or ``map``, or to apply a viewer axis swap from
    :data:`~sam3d_objects.integrations.t4.frames.VIEWER_AXES`.
    """
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trans = np.asarray(translation, dtype=np.float64).reshape(3)
    return BoxAlignment(
        linear=rot @ alignment.linear,
        translation=rot @ alignment.translation + trans,
        frame=frame,
        report=alignment.report,
    )


def _describe_axis_map(rotation) -> dict[str, str]:
    """Report which canonical object axis ended up on each box axis."""
    names = ("+X", "+Y", "+Z")
    out = {}
    for col, axis in enumerate(("forward(+X)", "left(+Y)", "up(+Z)")):
        # Column `col` of R^T is the box-frame image of object axis `col`;
        # we want the inverse question, so look along the rows.
        row = np.asarray(rotation, dtype=np.float64)[col]
        idx = int(np.argmax(np.abs(row)))
        sign = "+" if row[idx] >= 0 else "-"
        out[axis] = f"{sign}{names[idx][1:]}_obj"
    return out
