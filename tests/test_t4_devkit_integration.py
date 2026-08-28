"""Checks against the real ``t4-devkit`` v0.8.0 types, without needing a dataset.

The T4 conventions this bridge relies on -- ``Box3D.size`` being
``(width, length, height)``, the box frame being X-forward/Y-left/Z-up, and the
camera frame being OpenCV -- are asserted here against the library itself, so a
devkit upgrade that changes them fails loudly instead of silently rotating every
exported car.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

pytest.importorskip("t4_devkit")

from t4_devkit.common.geometry import view_points
from t4_devkit.dataclass import Box3D, SemanticLabel, Shape, ShapeType
from t4_devkit.typing import Quaternion

from sam3d_objects.integrations.t4.align import align_to_box, compose_alignment, robust_extent
from sam3d_objects.integrations.t4.dataset import CameraFrame, box_mask, project_points, select_boxes
from sam3d_objects.integrations.t4.frames import OPENCV_FROM_PYTORCH3D, matrix_to_quat, rotz

from .test_t4_alignment import (
    BOX_SIZE_WLH,
    CANONICAL_TO_BOX,
    box_pose_in_camera,
    canonical_car,
    fake_sam3d_output,
)


def make_box(rotation, position, size=BOX_SIZE_WLH, label="car", uuid="instance-0") -> Box3D:
    return Box3D(
        unix_time=0,
        frame_id="CAM_FRONT",
        semantic_label=SemanticLabel(label),
        position=tuple(np.asarray(position, dtype=float)),
        rotation=Quaternion(matrix_to_quat(rotation)),
        shape=Shape(shape_type=ShapeType.BOUNDING_BOX, size=tuple(float(v) for v in size)),
        uuid=uuid,
        num_points=500,
    )


def make_frame(boxes, width=1920, height=1080, focal=1200.0) -> CameraFrame:
    intrinsic = np.array(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return CameraFrame(
        sample_token="sample-0",
        sample_data_token="sample-data-0",
        channel="CAM_FRONT",
        image=np.zeros((height, width, 3), dtype=np.uint8),
        intrinsic=intrinsic,
        distortion=None,
        boxes=list(boxes),
        rot_ego_cam=np.eye(3),
        trans_ego_cam=np.zeros(3),
        rot_map_ego=np.eye(3),
        trans_map_ego=np.zeros(3),
    )


# --- conventions we depend on ----------------------------------------------


def test_box_size_is_width_length_height():
    box = make_box(np.eye(3), [0.0, 0.0, 10.0], size=(1.6, 4.5, 1.4))
    assert tuple(box.size) == (1.6, 4.5, 1.4)

    corners = box.corners() - np.asarray(box.position)
    extent = corners.max(axis=0) - corners.min(axis=0)
    # length spans the box's X axis, width its Y axis, height its Z axis
    assert extent == pytest.approx([4.5, 1.6, 1.4])


def test_box_rotation_matrix_maps_box_local_to_the_parent_frame():
    yaw = 0.73
    box = make_box(rotz(yaw), [3.0, 0.0, 12.0])
    forward = box.rotation.rotation_matrix @ np.array([1.0, 0.0, 0.0])
    assert forward == pytest.approx([np.cos(yaw), np.sin(yaw), 0.0])

    corners = box.corners()
    front = corners[:4].mean(axis=0) - np.asarray(box.position)
    assert front / np.linalg.norm(front) @ forward > 0.9  # first 4 corners face forward


def test_project_points_matches_view_points_without_distortion():
    frame = make_frame([])
    points = np.array([[0.0, 0.0, 10.0], [1.0, -0.5, 8.0], [-2.0, 1.0, 25.0]])
    mine = project_points(points, frame.intrinsic)
    theirs = view_points(points.T, frame.intrinsic, normalize=True)[:2].T
    assert mine == pytest.approx(theirs)


def test_project_points_drops_geometry_behind_the_camera():
    frame = make_frame([])
    assert project_points(np.array([[0.0, 0.0, -5.0]]), frame.intrinsic) is None
    both = project_points(np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]]), frame.intrinsic)
    assert both.shape == (1, 2)


# --- masks and selection ----------------------------------------------------


def test_hull_mask_covers_the_projected_box():
    box = make_box(rotz(0.3), [0.0, 0.0, 12.0])
    frame = make_frame([box])
    mask = box_mask(None, frame, box, source="hull")

    assert mask is not None and mask.any()
    uv = project_points(box.corners(), frame.intrinsic)
    rows, cols = np.nonzero(mask)
    assert cols.min() <= np.ceil(uv[:, 0].min())
    assert cols.max() >= np.floor(uv[:, 0].max()) - 1
    assert rows.min() <= np.ceil(uv[:, 1].min())


def test_hull_mask_returns_none_for_a_box_behind_the_camera():
    box = make_box(np.eye(3), [0.0, 0.0, -12.0])
    frame = make_frame([box])
    assert box_mask(None, frame, box, source="hull") is None


def test_select_boxes_filters_by_category_and_range():
    near_car = make_box(np.eye(3), [0.0, 0.0, 10.0], label="car", uuid="a")
    far_car = make_box(np.eye(3), [0.0, 0.0, 90.0], label="car", uuid="b")
    truck = make_box(np.eye(3), [0.0, 0.0, 10.0], label="vehicle.truck", uuid="c")
    frame = make_frame([near_car, far_car, truck])

    assert [b.uuid for b in select_boxes(frame, categories=["car"])] == ["a", "b"]
    assert [b.uuid for b in select_boxes(frame, categories=["car"], max_distance=50)] == ["a"]
    assert [b.uuid for b in select_boxes(frame, categories=["truck"])] == ["c"]
    assert len(select_boxes(frame)) == 3


# --- the whole chain --------------------------------------------------------


def test_reconstruction_fills_a_real_box3d_and_faces_forward():
    rot_cam_box, pos_cam_box = box_pose_in_camera(yaw=0.4)
    box = make_box(rot_cam_box, pos_cam_box)
    quat, trans, scale = fake_sam3d_output(
        rot_cam_box, pos_cam_box, yaw_error=np.deg2rad(8.0), scale_error=0.6
    )
    alignment = align_to_box(
        canonical_car(),
        sam3d_rotation=quat,
        sam3d_translation=trans,
        sam3d_scale=scale,
        box_rotation=box.rotation.rotation_matrix,
        box_position=np.asarray(box.position),
        box_size=np.asarray(box.size),
        percentile=0.0,
    )

    # the car's long axis ends up on the box's forward axis
    forward = alignment.linear @ np.array([0.0, 1.0, 0.0])
    assert forward / np.linalg.norm(forward) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert alignment.report["yaw_correction_deg"] == pytest.approx(-8.0, abs=1e-6)
    assert alignment.report["axis_map"]["forward(+X)"] == "+Y_obj"

    # and in the camera frame it fills exactly the annotated corners
    in_camera = compose_alignment(
        alignment, box.rotation.rotation_matrix, np.asarray(box.position), "camera"
    )
    aligned = canonical_car() @ in_camera.linear.T + in_camera.translation
    corners = box.corners()
    for axis in range(3):
        assert aligned[:, axis].min() == pytest.approx(corners[:, axis].min(), abs=1e-9)
        assert aligned[:, axis].max() == pytest.approx(corners[:, axis].max(), abs=1e-9)


def test_base_link_export_places_the_car_where_the_annotation_says():
    """base_link is X-forward/Y-left/Z-up, so a車 must come out heading along ego +X."""
    rot_cam_ego = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    yaw = 0.0  # the annotated car drives straight ahead
    rot_cam_box = rot_cam_ego @ rotz(yaw)
    pos_cam_box = rot_cam_ego @ np.array([18.0, 2.0, 0.6])

    box = make_box(rot_cam_box, pos_cam_box)
    frame = make_frame([box])
    frame.rot_ego_cam = rot_cam_ego.T
    frame.trans_ego_cam = np.zeros(3)

    quat, trans, scale = fake_sam3d_output(rot_cam_box, pos_cam_box, yaw_error=np.deg2rad(5.0))
    alignment = align_to_box(
        canonical_car(),
        sam3d_rotation=quat,
        sam3d_translation=trans,
        sam3d_scale=scale,
        box_rotation=box.rotation.rotation_matrix,
        box_position=np.asarray(box.position),
        box_size=np.asarray(box.size),
        percentile=0.0,
    )
    in_camera = compose_alignment(
        alignment, box.rotation.rotation_matrix, np.asarray(box.position), "camera"
    )
    in_ego = compose_alignment(in_camera, frame.rot_ego_cam, frame.trans_ego_cam, "base_link")

    aligned = canonical_car() @ in_ego.linear.T + in_ego.translation
    lower, upper, extent = robust_extent(aligned, percentile=0.0)
    assert (lower + upper) / 2 == pytest.approx([18.0, 2.0, 0.6], abs=1e-9)
    # heading along ego +X means length on X and width on Y
    assert extent == pytest.approx([BOX_SIZE_WLH[1], BOX_SIZE_WLH[0], BOX_SIZE_WLH[2]], rel=1e-9)

    forward = in_ego.linear @ np.array([0.0, 1.0, 0.0])
    assert forward / np.linalg.norm(forward) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_forgetting_the_camera_flip_turns_the_car_upside_down():
    """A front camera looks along the ground, so the missed half turn flips "up"."""
    rot_cam_ego = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    rot_cam_box = rot_cam_ego @ rotz(0.0)
    pos_cam_box = rot_cam_ego @ np.array([18.0, 0.0, 0.6])
    box = make_box(rot_cam_box, pos_cam_box)
    quat, trans, scale = fake_sam3d_output(rot_cam_box, pos_cam_box)

    from sam3d_objects.integrations.t4.frames import sam3d_pose_to_rigid

    rot_p3d_obj, _, _ = sam3d_pose_to_rigid(quat, trans, scale)
    rot_cam_box = box.rotation.rotation_matrix
    correct = rot_cam_box.T @ OPENCV_FROM_PYTORCH3D @ rot_p3d_obj
    naive = rot_cam_box.T @ rot_p3d_obj

    assert correct == pytest.approx(CANONICAL_TO_BOX, abs=1e-9)
    assert correct @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    assert naive @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
