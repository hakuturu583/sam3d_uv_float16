"""Coordinate-frame and box-fitting tests for the T4 <-> SAM 3D bridge.

These run on CPU without a GPU, a checkpoint or a T4 dataset: the SAM 3D layout
output is synthesised from a known ground truth, perturbed the way the model
would be wrong, and the aligner is asked to recover it.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

from sam3d_objects.integrations.t4.align import align_to_box, compose_alignment, robust_extent
from sam3d_objects.integrations.t4.frames import (
    OPENCV_FROM_PYTORCH3D,
    VIEWER_AXES,
    axis_aligned_rotations,
    matrix_to_quat,
    obj_to_box_pose,
    quat_to_matrix,
    rotz,
    sam3d_pose_to_rigid,
    snap_to_axis_rotation,
    snap_yaw_only,
    yaw_of,
)

# --- synthetic ground truth -------------------------------------------------
#
# A "car" in the SAM 3D canonical object frame whose長手方向 is +Y_obj and whose
# roof is +Z_obj. `CANONICAL_TO_BOX` is then the rotation SAM 3D would have to
# predict for the object to sit square in a T4 box (+X forward, +Y left, +Z up).
HALF_EXTENT_OBJ = np.array([0.2, 0.45, 0.15])
CANONICAL_TO_BOX = rotz(-np.pi / 2)  # +Y_obj -> +X_box, +Z_obj -> +Z_box
TRUE_SCALE = 4.0
BOX_SIZE_WLH = (
    2 * HALF_EXTENT_OBJ[0] * TRUE_SCALE,  # width  <- X_obj
    2 * HALF_EXTENT_OBJ[1] * TRUE_SCALE,  # length <- Y_obj
    2 * HALF_EXTENT_OBJ[2] * TRUE_SCALE,  # height <- Z_obj
)


def canonical_car(n: int = 4000, seed: int = 0) -> np.ndarray:
    """Points filling the canonical object box, as `Gaussian.get_xyz` would.

    The 8 corners are included so the sampled extent is exactly the box extent
    and the fitted scale can be asserted without sampling slack.
    """
    rng = np.random.default_rng(seed)
    interior = rng.uniform(-1.0, 1.0, size=(n, 3)) * HALF_EXTENT_OBJ
    corners = np.array(list(np.ndindex(2, 2, 2)), dtype=float) * 2.0 - 1.0
    return np.vstack([interior, corners * HALF_EXTENT_OBJ])


def box_pose_in_camera(yaw: float = 0.6, pitch: float = 0.0):
    """A plausible T4 box pose in the OpenCV camera frame.

    A camera looks down +Z with +Y down, so a vehicle on the ground plane has its
    box "up" axis pointing at -Y_cam.
    """
    # ego(+X fwd, +Y left, +Z up) as seen from an OpenCV camera(+X right, +Y down, +Z fwd)
    rot_cam_ego = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    rot_ego_box = rotz(yaw) @ np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    return rot_cam_ego @ rot_ego_box, np.array([1.5, 0.4, 12.0])


def fake_sam3d_output(rot_cam_box, pos_cam_box, *, yaw_error=0.0, tilt=0.0, scale_error=1.0):
    """Build the layout output SAM 3D would emit for a given (perturbed) pose.

    Returns the ``(rotation, translation, scale)`` triple in exactly the form
    ``Inference.__call__`` produces: a ``(w, x, y, z)`` quaternion in the
    PyTorch3D camera frame, applied through PyTorch3D's row-vector convention.
    """
    perturb = rotz(yaw_error) @ np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(tilt), -np.sin(tilt)],
            [0.0, np.sin(tilt), np.cos(tilt)],
        ]
    )
    rot_cam_obj = rot_cam_box @ perturb @ CANONICAL_TO_BOX
    rot_p3d_obj = OPENCV_FROM_PYTORCH3D @ rot_cam_obj
    trans_p3d = OPENCV_FROM_PYTORCH3D @ pos_cam_box
    # sam3d_pose_to_rigid() transposes, so undo that here.
    return matrix_to_quat(rot_p3d_obj.T), trans_p3d, TRUE_SCALE * scale_error


# --- frame conventions ------------------------------------------------------


def test_pytorch3d_to_opencv_is_a_180_degree_roll():
    assert np.allclose(OPENCV_FROM_PYTORCH3D, np.diag([-1.0, -1.0, 1.0]))
    assert np.isclose(np.linalg.det(OPENCV_FROM_PYTORCH3D), 1.0)
    # involutive: applying it twice is a no-op
    assert np.allclose(OPENCV_FROM_PYTORCH3D @ OPENCV_FROM_PYTORCH3D, np.eye(3))


def test_sam3d_pose_uses_the_row_vector_convention():
    """`make_scene` does `points @ R`; the column-vector form must transpose it."""
    quat = matrix_to_quat(rotz(0.7))
    rot, trans, scale = sam3d_pose_to_rigid(quat, [1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    points = np.random.default_rng(1).normal(size=(50, 3))

    pytorch3d_style = scale * (points @ quat_to_matrix(quat)) + trans
    column_style = (scale * (rot @ points.T)).T + trans
    assert np.allclose(pytorch3d_style, column_style)


def test_anisotropic_scale_warns():
    with pytest.warns(RuntimeWarning):
        sam3d_pose_to_rigid([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0])


def test_obj_to_box_recovers_the_exact_convention_rotation():
    """With a perfect prediction the residual must be exactly CANONICAL_TO_BOX."""
    rot_cam_box, pos_cam_box = box_pose_in_camera()
    quat, trans, scale = fake_sam3d_output(rot_cam_box, pos_cam_box)

    rot_box_obj, trans_box, recovered_scale = obj_to_box_pose(
        sam3d_rotation=quat,
        sam3d_translation=trans,
        sam3d_scale=scale,
        box_rotation=rot_cam_box,
        box_position=pos_cam_box,
    )
    assert np.allclose(rot_box_obj, CANONICAL_TO_BOX, atol=1e-9)
    assert np.allclose(trans_box, 0.0, atol=1e-9)
    assert np.isclose(recovered_scale, TRUE_SCALE)


def test_dropping_the_camera_flip_rolls_the_object_by_180_degrees():
    """Guard the pitfall this module exists for: p3d and OpenCV differ by 180 deg.

    Skipping the flip post-multiplies by a half turn about the camera's optical
    axis. A driving camera looks roughly along the ground, so the symptom is a
    car that comes out upside down rather than one that is merely mis-scaled.
    """
    rot_cam_box, pos_cam_box = box_pose_in_camera()
    quat, trans, scale = fake_sam3d_output(rot_cam_box, pos_cam_box)
    rot_p3d_obj, _, _ = sam3d_pose_to_rigid(quat, trans, scale)

    naive = rot_cam_box.T @ rot_p3d_obj  # forgot OPENCV_FROM_PYTORCH3D
    correct = rot_cam_box.T @ OPENCV_FROM_PYTORCH3D @ rot_p3d_obj
    assert np.allclose(correct, CANONICAL_TO_BOX, atol=1e-9)

    error = naive @ correct.T
    assert np.trace(error) == pytest.approx(-1.0, abs=1e-9)  # a half turn
    optical_axis_in_box = rot_cam_box.T @ np.array([0.0, 0.0, 1.0])
    assert error @ optical_axis_in_box == pytest.approx(optical_axis_in_box, abs=1e-9)

    assert correct @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert naive @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


# --- rotation snapping ------------------------------------------------------


def test_axis_aligned_rotations_are_24_distinct_proper_rotations():
    rotations = list(axis_aligned_rotations())
    assert len(rotations) == 24
    for rot in rotations:
        assert np.isclose(np.linalg.det(rot), 1.0)
        assert np.allclose(rot @ rot.T, np.eye(3))
    flattened = {tuple(np.round(r.ravel(), 6)) for r in rotations}
    assert len(flattened) == 24


@pytest.mark.parametrize("error_deg", [-11.0, -3.0, 0.0, 4.5, 12.0])
def test_snap_removes_a_small_heading_error(error_deg):
    perturbed = rotz(np.deg2rad(error_deg)) @ CANONICAL_TO_BOX
    snapped, angle = snap_to_axis_rotation(perturbed)
    assert np.allclose(snapped, CANONICAL_TO_BOX, atol=1e-9)
    assert np.isclose(np.degrees(angle), abs(error_deg), atol=1e-6)


def test_yaw_mode_keeps_roll_and_pitch():
    tilt = np.deg2rad(8.0)
    roll = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]]
    )
    perturbed = rotz(np.deg2rad(5.0)) @ roll @ CANONICAL_TO_BOX

    corrected, delta = snap_yaw_only(perturbed)
    assert np.degrees(delta) == pytest.approx(-5.0, abs=1e-9)
    # the roll survives, so this is *not* an axis-aligned rotation
    assert not np.allclose(corrected, CANONICAL_TO_BOX, atol=1e-3)
    assert np.linalg.norm(corrected @ np.array([0.0, 0.0, 1.0]) - [0.0, 0.0, 1.0]) > 1e-3

    # ...but the forward axis now projects exactly onto +X.
    forward = corrected @ np.array([0.0, 1.0, 0.0])
    assert np.degrees(np.arctan2(forward[1], forward[0])) == pytest.approx(0.0, abs=1e-9)
    assert yaw_of(rotz(np.deg2rad(5.0))) == pytest.approx(np.deg2rad(5.0))


# --- box fitting ------------------------------------------------------------


def test_robust_extent_ignores_outliers():
    points = np.zeros((1000, 3))
    points[:, 0] = np.linspace(-1.0, 1.0, 1000)
    points[0, 0] = -50.0  # a stray splat
    _, _, extent = robust_extent(points, percentile=1.0)
    assert extent[0] == pytest.approx(2.0, abs=0.05)


def align_perturbed(
    *, box_size=BOX_SIZE_WLH, yaw_error=0.0, tilt=0.0, scale_error=1.0, **align_kwargs
):
    """Align a deliberately-wrong SAM 3D output onto a box, and hand back both."""
    rot_cam_box, pos_cam_box = box_pose_in_camera()
    quat, trans, scale = fake_sam3d_output(
        rot_cam_box, pos_cam_box, yaw_error=yaw_error, tilt=tilt, scale_error=scale_error
    )
    alignment = align_to_box(
        canonical_car(),
        sam3d_rotation=quat,
        sam3d_translation=trans,
        sam3d_scale=scale,
        box_rotation=rot_cam_box,
        box_position=pos_cam_box,
        box_size=box_size,
        percentile=0.0,
        **align_kwargs,
    )
    return alignment, rot_cam_box, pos_cam_box


def test_alignment_makes_the_car_face_forward():
    yaw_error_deg = 9.0
    alignment, _, _ = align_perturbed(yaw_error=np.deg2rad(yaw_error_deg), scale_error=0.8)

    # the canonical長手方向 (+Y_obj) must come out along the box's forward axis
    forward = alignment.linear @ np.array([0.0, 1.0, 0.0])
    forward /= np.linalg.norm(forward)
    assert np.allclose(forward, [1.0, 0.0, 0.0], atol=1e-9)

    up = alignment.linear @ np.array([0.0, 0.0, 1.0])
    up /= np.linalg.norm(up)
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-9)

    assert alignment.report["yaw_correction_deg"] == pytest.approx(-yaw_error_deg, abs=1e-6)
    assert alignment.report["snap_angle_deg"] == pytest.approx(yaw_error_deg, abs=1e-6)


def test_alignment_restores_the_metric_size():
    alignment, _, _ = align_perturbed(yaw_error=np.deg2rad(6.0), scale_error=0.55)
    aligned = canonical_car() @ alignment.linear.T + alignment.translation

    width, length, height = BOX_SIZE_WLH
    lower, upper, extent = robust_extent(aligned, percentile=0.0)
    assert extent == pytest.approx([length, width, height], rel=1e-9)
    assert (lower + upper) / 2 == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    # SAM 3D was 45% too small; the report says so.
    assert np.mean(alignment.report["scale_ratio_vs_sam3d"]) == pytest.approx(1 / 0.55, rel=1e-6)


def test_box_size_is_width_length_height_not_xyz():
    """A wrong `size` order silently swaps a car's length and width."""
    alignment, _, _ = align_perturbed()
    assert alignment.report["target_size_xyz"] == pytest.approx(
        [BOX_SIZE_WLH[1], BOX_SIZE_WLH[0], BOX_SIZE_WLH[2]]
    )


def test_bottom_alignment_seats_the_object_on_the_box_floor():
    alignment, _, _ = align_perturbed(z_align="bottom")
    aligned = canonical_car() @ alignment.linear.T + alignment.translation
    assert aligned[:, 2].min() == pytest.approx(-BOX_SIZE_WLH[2] / 2, abs=1e-9)


def test_axis_scale_mode_fills_a_box_of_a_different_aspect_ratio():
    """`axis` trades shape fidelity for an exact fill; `iso` would leave a gap."""
    # A box that is wider and flatter than the reconstruction.
    stretched = (BOX_SIZE_WLH[0] * 1.3, BOX_SIZE_WLH[1], BOX_SIZE_WLH[2] * 0.8)
    kwargs = dict(box_size=stretched, scale_error=0.7)
    anisotropic, _, _ = align_perturbed(scale_mode="axis", **kwargs)
    isotropic, _, _ = align_perturbed(scale_mode="iso", **kwargs)

    width, length, height = stretched
    aligned = canonical_car() @ anisotropic.linear.T + anisotropic.translation
    _, _, extent = robust_extent(aligned, percentile=0.0)
    assert extent == pytest.approx([length, width, height], rel=1e-9)
    assert not anisotropic.is_similarity

    iso_aligned = canonical_car() @ isotropic.linear.T + isotropic.translation
    _, _, iso_extent = robust_extent(iso_aligned, percentile=0.0)
    assert isotropic.is_similarity
    assert iso_extent[1] < width  # the isotropic fit cannot fill the wider box


def test_composing_to_the_camera_frame_lands_inside_the_box():
    alignment, rot_cam_box, pos_cam_box = align_perturbed(yaw_error=np.deg2rad(7.0))
    in_camera = compose_alignment(alignment, rot_cam_box, pos_cam_box, "camera")

    aligned = canonical_car() @ in_camera.linear.T + in_camera.translation

    # back-project into the box frame: the object must fill the annotation exactly
    local = (aligned - pos_cam_box) @ rot_cam_box
    width, length, height = BOX_SIZE_WLH
    half = np.array([length, width, height]) / 2
    assert np.all(np.abs(local) <= half + 1e-9)
    assert local.max(axis=0) == pytest.approx(half, rel=1e-9)
    assert local.min(axis=0) == pytest.approx(-half, rel=1e-9)


def test_rotation_mode_none_keeps_the_predicted_error():
    alignment, _, _ = align_perturbed(yaw_error=np.deg2rad(10.0), rotation_mode="none")
    forward = alignment.linear @ np.array([0.0, 1.0, 0.0])
    forward /= np.linalg.norm(forward)
    assert np.degrees(np.arctan2(forward[1], forward[0])) == pytest.approx(10.0, abs=1e-6)


def test_gltf_viewer_axes_put_forward_on_minus_z():
    gltf = VIEWER_AXES["gltf"]
    assert np.isclose(np.linalg.det(gltf), 1.0)
    assert np.allclose(gltf @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0])  # forward -> -Z
    assert np.allclose(gltf @ np.array([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0])  # up -> +Y


@pytest.mark.parametrize(
    ("option", "value"),
    [("rotation_mode", "nope"), ("scale_mode", "nope"), ("z_align", "nope")],
)
def test_invalid_modes_are_rejected(option, value):
    with pytest.raises(ValueError, match=option):
        align_perturbed(**{option: value})


def test_extra_yaw_turns_a_back_to_front_car_around():
    """SAM 3D sometimes reads a symmetric vehicle backwards; 180 fixes it."""
    straight, _, _ = align_perturbed()
    flipped, _, _ = align_perturbed(extra_yaw_deg=180.0)

    forward = np.array([0.0, 1.0, 0.0])
    assert straight.linear @ forward == pytest.approx(-(flipped.linear @ forward), abs=1e-9)
    assert flipped.report["extra_yaw_deg"] == 180.0
    # the fit is unchanged otherwise
    assert flipped.report["fitted_scale"] == pytest.approx(straight.report["fitted_scale"])
