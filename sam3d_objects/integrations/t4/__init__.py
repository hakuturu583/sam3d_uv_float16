"""Bridge between SAM 3D Objects and the T4 dataset (tier4/t4-devkit v0.8.0).

The sub-modules are layered so that the coordinate math can be imported and
tested without CUDA, ``torch`` or ``t4-devkit`` being installed:

* :mod:`~sam3d_objects.integrations.t4.frames` -- pure ``numpy`` frame conventions.
* :mod:`~sam3d_objects.integrations.t4.align`  -- pure ``numpy`` box alignment solver.
* :mod:`~sam3d_objects.integrations.t4.gaussian_ops` -- ``torch`` ops on ``Gaussian``.
* :mod:`~sam3d_objects.integrations.t4.dataset` -- ``t4-devkit`` loading helpers.
"""

from .align import BoxAlignment, align_to_box, compose_alignment
from .frames import (
    OPENCV_FROM_PYTORCH3D,
    PYTORCH3D_FROM_OPENCV,
    VIEWER_AXES,
    axis_aligned_rotations,
    matrix_to_quat,
    obj_to_box_pose,
    quat_multiply,
    quat_to_matrix,
    sam3d_pose_to_rigid,
    snap_to_axis_rotation,
    yaw_of,
)

__all__ = [
    "OPENCV_FROM_PYTORCH3D",
    "PYTORCH3D_FROM_OPENCV",
    "VIEWER_AXES",
    "BoxAlignment",
    "align_to_box",
    "axis_aligned_rotations",
    "compose_alignment",
    "matrix_to_quat",
    "obj_to_box_pose",
    "quat_multiply",
    "quat_to_matrix",
    "sam3d_pose_to_rigid",
    "snap_to_axis_rotation",
    "yaw_of",
]
