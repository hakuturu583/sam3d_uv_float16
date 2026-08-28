"""Coordinate frame conventions shared by SAM 3D Objects and the T4 dataset.

Frames
------
``obj``
    SAM 3D Objects canonical object frame. ``Gaussian.get_xyz`` lives in the
    ``[-0.5, 0.5]^3`` cube (see ``SLatGaussianDecoder.to_representation``,
    ``aabb=[-0.5, -0.5, -0.5, 1, 1, 1]``). It is Z-up: the GLB exporter rotates
    it "from z-up to y-up" before writing glTF.
``p3d``
    PyTorch3D camera frame: **+X left, +Y up, +Z forward**, origin at the optical
    centre. This is the frame the SAM 3D layout head predicts into -- the MoGe
    point map (OpenCV convention) is rotated into it by
    ``camera_to_pytorch3d_camera`` in ``inference_pipeline_pointmap.py``.
``cam``
    T4 camera sensor frame, OpenCV convention: **+X right, +Y down, +Z forward**.
    ``t4_devkit`` projects with ``u = K @ p`` directly, so this is what
    ``T4Devkit.get_sample_data(..., as_sensor_coord=True)`` returns boxes in.
``box``
    T4 3D bounding box local frame: **+X forward (length), +Y left (width),
    +Z up (height)** -- see ``Box3D.corners``. ``Box3D.size`` is ordered
    ``(width, length, height)``.
``ego``
    ``base_link``: +X forward, +Y left, +Z up.

The single non-obvious conversion is ``p3d -> cam``, which is a 180 degree roll
about the optical axis: X and Y flip, Z (depth) is shared.
"""

from __future__ import annotations

import warnings
from typing import Iterator

import numpy as np

__all__ = [
    "OPENCV_FROM_PYTORCH3D",
    "PYTORCH3D_FROM_OPENCV",
    "VIEWER_AXES",
    "axis_aligned_rotations",
    "heading_correction",
    "matrix_to_quat",
    "obj_to_box_pose",
    "quat_multiply",
    "quat_to_matrix",
    "rotz",
    "sam3d_pose_to_rigid",
    "snap_to_axis_rotation",
    "snap_yaw_only",
    "yaw_of",
]


#: ``p_cam = OPENCV_FROM_PYTORCH3D @ p_p3d``. The matrix is its own inverse.
OPENCV_FROM_PYTORCH3D = np.diag([-1.0, -1.0, 1.0])

#: ``p_p3d = PYTORCH3D_FROM_OPENCV @ p_cam``.
PYTORCH3D_FROM_OPENCV = OPENCV_FROM_PYTORCH3D

#: Extra axis swaps for third party viewers, applied to points already expressed
#: in a T4-style frame (+X forward, +Y left, +Z up).
#:
#: ``gltf`` is the glTF/three.js convention (+X right, +Y up, -Z forward), which
#: is what most Gaussian-splat web viewers assume.
VIEWER_AXES = {
    "none": np.eye(3),
    "gltf": np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ]
    ),
}


def quat_to_matrix(q) -> np.ndarray:
    """Return the column-vector rotation matrix of a ``(w, x, y, z)`` quaternion.

    Matches ``pytorch3d.transforms.quaternion_to_matrix`` and the 3DGS
    ``build_rotation`` helper, i.e. ``p_rotated = R @ p``.
    """
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("cannot normalize a zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(matrix) -> np.ndarray:
    """Return the ``(w, x, y, z)`` quaternion of a column-vector rotation matrix."""
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    q /= np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def quat_multiply(a, b) -> np.ndarray:
    """Hamilton product of two ``(w, x, y, z)`` quaternions."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def rotz(angle: float) -> np.ndarray:
    """Rotation of ``angle`` radians about +Z (yaw)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of(matrix) -> float:
    """Return the yaw (rotation about +Z) of a rotation matrix, in radians."""
    m = np.asarray(matrix, dtype=np.float64)
    return float(np.arctan2(m[1, 0], m[0, 0]))


def sam3d_pose_to_rigid(rotation, translation, scale):
    """Convert a SAM 3D layout prediction into a column-vector similarity.

    SAM 3D returns ``rotation`` (quaternion, ``w x y z``), ``translation`` and an
    isotropic ``scale`` that ``make_scene`` applies through PyTorch3D's
    *row-vector* convention (``p @ R``). This returns ``(R, t, s)`` such that::

        p_p3d = s * R @ p_obj + t

    which is the usual column-vector form. Note the transpose: it is the single
    easiest place to introduce a mirrored/rotated reconstruction.

    Args:
        rotation: ``(4,)`` quaternion in ``(w, x, y, z)`` order (``output["rotation"]``).
        translation: ``(3,)`` translation (``output["translation"]``).
        scale: scalar or ``(3,)`` isotropic scale (``output["scale"]``).

    Returns:
        ``(R, t, s)`` with ``R`` a ``(3, 3)`` rotation, ``t`` a ``(3,)`` vector and
        ``s`` a float.
    """
    quat = np.asarray(rotation, dtype=np.float64).reshape(-1)[:4]
    rot = quat_to_matrix(quat).T  # row-vector (p @ R) -> column-vector (R^T @ p)
    trans = np.asarray(translation, dtype=np.float64).reshape(-1)[:3]

    scale_arr = np.asarray(scale, dtype=np.float64).reshape(-1)
    if scale_arr.size > 1 and not np.allclose(scale_arr, scale_arr[0], rtol=1e-5, atol=1e-8):
        warnings.warn(
            f"SAM 3D returned an anisotropic scale {scale_arr}; using its mean. "
            "The pipeline normally collapses it to a single isotropic value.",
            RuntimeWarning,
            stacklevel=2,
        )
    return rot, trans, float(scale_arr.mean())


def obj_to_box_pose(
    *,
    sam3d_rotation,
    sam3d_translation,
    sam3d_scale,
    box_rotation,
    box_position,
):
    """Express the SAM 3D reconstruction in the T4 bounding box local frame.

    Chains ``obj -> p3d -> cam -> box``.

    Args:
        sam3d_rotation: ``output["rotation"]``, quaternion ``(w, x, y, z)``.
        sam3d_translation: ``output["translation"]``.
        sam3d_scale: ``output["scale"]``.
        box_rotation: ``(3, 3)`` ``Box3D.rotation.rotation_matrix`` of a box that
            was retrieved with ``as_sensor_coord=True`` for this camera, i.e. it
            maps box-local points into the OpenCV camera frame.
        box_position: ``(3,)`` ``Box3D.position`` in the same camera frame.

    Returns:
        ``(R_box_obj, t_box, s)`` with ``p_box = s * R_box_obj @ p_obj + t_box``.
    """
    rot_p3d_obj, trans_p3d, scale = sam3d_pose_to_rigid(
        sam3d_rotation, sam3d_translation, sam3d_scale
    )

    rot_cam_obj = OPENCV_FROM_PYTORCH3D @ rot_p3d_obj
    trans_cam = OPENCV_FROM_PYTORCH3D @ trans_p3d

    rot_cam_box = np.asarray(box_rotation, dtype=np.float64).reshape(3, 3)
    pos_cam_box = np.asarray(box_position, dtype=np.float64).reshape(3)

    rot_box_obj = rot_cam_box.T @ rot_cam_obj
    trans_box = rot_cam_box.T @ (trans_cam - pos_cam_box)
    return rot_box_obj, trans_box, scale


def axis_aligned_rotations() -> Iterator[np.ndarray]:
    """Yield the 24 proper rotations that map the coordinate axes onto themselves."""
    eye = np.eye(3)
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in np.ndindex(2, 2, 2):
            sign_vec = 1.0 - 2.0 * np.asarray(signs, dtype=np.float64)
            candidate = (eye[list(perm)] * sign_vec[:, None]).T
            if np.linalg.det(candidate) > 0.0:
                yield candidate


def snap_to_axis_rotation(matrix):
    """Snap a rotation to the closest axis-aligned rotation (Frobenius distance).

    SAM 3D's layout head gets the *discrete* pose right far more often than the
    last few degrees of it, so snapping removes the residual error while keeping
    the front/back and up/down decision the model made from the image.

    Returns:
        ``(R_snapped, angle)`` where ``angle`` is the geodesic distance in radians
        between the input and the snapped rotation.
    """
    rot = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    best, best_trace = None, -np.inf
    for candidate in axis_aligned_rotations():
        trace = float(np.trace(candidate.T @ rot))
        if trace > best_trace:
            best, best_trace = candidate, trace
    angle = float(np.arccos(np.clip((best_trace - 1.0) / 2.0, -1.0, 1.0)))
    return best, angle


def snap_yaw_only(matrix):
    """Snap only the heading, keeping the roll/pitch SAM 3D estimated.

    Useful when the object genuinely is not level (a car on a slope, a tilted
    parked bike) and only the yaw should be forced onto the box.

    The forward axis is whichever object axis :func:`snap_to_axis_rotation` would
    have sent to +X; the returned yaw is the exact rotation about +Z that puts
    that axis' ground projection back on +X.

    Returns:
        ``(R_corrected, delta_yaw)`` with ``delta_yaw`` in radians.
    """
    rot = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    target, _ = snap_to_axis_rotation(rot)
    forward_obj = target[0]  # object axis that should end up on +X
    forward = rot @ forward_obj
    if np.hypot(forward[0], forward[1]) < 1e-8:
        # The forward axis points (nearly) straight up; there is no meaningful
        # heading to read off it, so fall back to the full correction's yaw.
        delta = yaw_of(target @ rot.T)
    else:
        delta = -float(np.arctan2(forward[1], forward[0]))
    return rotz(delta) @ rot, float(delta)


def heading_correction(matrix) -> float:
    """Return, in radians, the yaw correction :func:`snap_yaw_only` would apply."""
    return snap_yaw_only(matrix)[1]
