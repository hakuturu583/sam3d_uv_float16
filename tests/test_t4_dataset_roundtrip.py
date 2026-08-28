"""End-to-end check on a synthetic but physically consistent T4 dataset.

Builds a one-frame dataset on disk with a realistic front-camera extrinsic (the
sample fixture shipped with ``t4-devkit`` uses an identity camera rotation, which
would not project), loads it through ``t4-devkit``, and runs the full
image -> box -> SAM 3D -> aligned splats chain with a synthesised model output.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

pytest.importorskip("t4_devkit")
from PIL import Image
from pyquaternion import Quaternion

from sam3d_objects.integrations.t4.align import align_to_box, compose_alignment, robust_extent
from sam3d_objects.integrations.t4.dataset import (
    box_mask,
    load_camera_frame,
    load_t4,
    project_points,
    select_boxes,
)
from sam3d_objects.integrations.t4.frames import matrix_to_quat, rotz

from .test_t4_alignment import BOX_SIZE_WLH, canonical_car, fake_sam3d_output

IMAGE_SIZE = (960, 540)  # (width, height)
FOCAL = 900.0

# A front camera mounted 1.6 m ahead of and 1.5 m above base_link. Its OpenCV
# axes (X right, Y down, Z forward) expressed in ego coordinates:
ROT_EGO_CAM = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)
TRANS_EGO_CAM = np.array([1.6, 0.0, 1.5])

# The annotated car, in base_link.
CAR_IN_EGO = np.array([16.0, 1.2, 0.9])
CAR_YAW = 0.18

# A non-trivial ego pose so the map frame is exercised too.
EGO_YAW = 0.35
TRANS_MAP_EGO = np.array([120.0, -45.0, 3.0])


def _tokens(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index:032d}"[-32:] for index in range(count)]


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory) -> str:
    """Write a single-sample T4 dataset with one annotated car ahead of the ego."""
    root = tmp_path_factory.mktemp("t4dataset")
    annotation = root / "annotation"
    annotation.mkdir()
    (root / "data" / "CAM_FRONT").mkdir(parents=True)

    width, height = IMAGE_SIZE
    Image.fromarray(np.full((height, width, 3), 96, dtype=np.uint8)).save(
        root / "data" / "CAM_FRONT" / "000.jpg"
    )

    (
        sensor,
        calib,
        ego_pose,
        log,
        scene,
        sample,
        sample_data,
        instance,
        category,
        attribute,
        visibility,
        annotation_3d,
        map_token,
    ) = _tokens("t", 13)

    rot_map_ego = rotz(EGO_YAW)
    car_in_map = rot_map_ego @ CAR_IN_EGO + TRANS_MAP_EGO
    car_rot_map = rot_map_ego @ rotz(CAR_YAW)

    tables = {
        "sensor": [{"token": sensor, "channel": "CAM_FRONT", "modality": "camera"}],
        "calibrated_sensor": [
            {
                "token": calib,
                "sensor_token": sensor,
                "translation": TRANS_EGO_CAM.tolist(),
                "rotation": matrix_to_quat(ROT_EGO_CAM).tolist(),
                "camera_intrinsic": [
                    [FOCAL, 0.0, width / 2],
                    [0.0, FOCAL, height / 2],
                    [0.0, 0.0, 1.0],
                ],
                "camera_distortion": [],
            }
        ],
        "ego_pose": [
            {
                "token": ego_pose,
                "timestamp": 1704067200000000,
                "translation": TRANS_MAP_EGO.tolist(),
                "rotation": matrix_to_quat(rot_map_ego).tolist(),
            }
        ],
        "log": [
            {
                "token": log,
                "logfile": "synthetic.bag",
                "vehicle": "test",
                "data_captured": "2024-01-01",
                "location": "test",
            }
        ],
        "map": [
            {
                "token": map_token,
                "log_tokens": [log],
                "category": "semantic_prior",
                "filename": "map/none.osm",
            }
        ],
        "scene": [
            {
                "token": scene,
                "name": "synthetic",
                "description": "one car ahead",
                "log_token": log,
                "nbr_samples": 1,
                "first_sample_token": sample,
                "last_sample_token": sample,
            }
        ],
        "sample": [
            {
                "token": sample,
                "timestamp": 1704067200000000,
                "scene_token": scene,
                "next": "",
                "prev": "",
            }
        ],
        "sample_data": [
            {
                "token": sample_data,
                "sample_token": sample,
                "ego_pose_token": ego_pose,
                "calibrated_sensor_token": calib,
                "filename": "data/CAM_FRONT/000.jpg",
                "fileformat": "jpg",
                "width": width,
                "height": height,
                "timestamp": 1704067200000000,
                "is_key_frame": True,
                "next": "",
                "prev": "",
            }
        ],
        "category": [{"token": category, "name": "car", "description": "a car"}],
        "attribute": [{"token": attribute, "name": "stopped", "description": "not moving"}],
        "visibility": [{"token": visibility, "level": "full", "description": "fully visible"}],
        "instance": [
            {
                "token": instance,
                "category_token": category,
                "instance_name": "car_000",
                "nbr_annotations": 1,
                "first_annotation_token": annotation_3d,
                "last_annotation_token": annotation_3d,
            }
        ],
        "sample_annotation": [
            {
                "token": annotation_3d,
                "sample_token": sample,
                "instance_token": instance,
                "attribute_tokens": [attribute],
                "visibility_token": visibility,
                "translation": car_in_map.tolist(),
                "size": list(BOX_SIZE_WLH),
                "rotation": Quaternion(matrix_to_quat(car_rot_map)).elements.tolist(),
                "num_lidar_pts": 320,
                "num_radar_pts": 0,
                "next": "",
                "prev": "",
            }
        ],
    }
    for name, rows in tables.items():
        (annotation / f"{name}.json").write_text(json.dumps(rows))
    return str(root)


@pytest.fixture(scope="module")
def frame(dataset_root):
    t4 = load_t4(dataset_root, verbose=False)
    return t4, load_camera_frame(t4, sample_index=0, channel="CAM_FRONT")


def test_boxes_come_back_in_the_opencv_camera_frame(frame):
    _, cam = frame
    assert len(cam.boxes) == 1
    box = cam.boxes[0]

    expected = ROT_EGO_CAM.T @ (CAR_IN_EGO - TRANS_EGO_CAM)
    assert np.asarray(box.position) == pytest.approx(expected, abs=1e-9)
    assert box.position[2] > 0  # +Z is depth, so the car is in front
    assert box.position[2] == pytest.approx(CAR_IN_EGO[0] - TRANS_EGO_CAM[0], abs=1e-9)

    uv = project_points(box.corners(), cam.intrinsic)
    width, height = cam.size
    assert np.all((uv[:, 0] > 0) & (uv[:, 0] < width))
    assert np.all((uv[:, 1] > 0) & (uv[:, 1] < height))


def test_extrinsics_round_trip_back_to_base_link(frame):
    _, cam = frame
    box = cam.boxes[0]
    in_ego = cam.rot_ego_cam @ np.asarray(box.position) + cam.trans_ego_cam
    assert in_ego == pytest.approx(CAR_IN_EGO, abs=1e-9)

    in_map = cam.rot_map_ego @ in_ego + cam.trans_map_ego
    assert in_map == pytest.approx(rotz(EGO_YAW) @ CAR_IN_EGO + TRANS_MAP_EGO, abs=1e-9)


def test_hull_mask_lands_on_the_car(frame):
    t4, cam = frame
    box = select_boxes(cam, categories=["car"])[0]
    mask = box_mask(t4, cam, box, source="hull")

    assert mask is not None and mask.any()
    uv = project_points(box.corners(), cam.intrinsic)
    rows, cols = np.nonzero(mask)
    assert cols.min() == pytest.approx(uv[:, 0].min(), abs=2)
    assert cols.max() == pytest.approx(uv[:, 0].max(), abs=2)
    assert rows.min() == pytest.approx(uv[:, 1].min(), abs=2)


def test_full_chain_puts_the_car_on_the_annotation_facing_forward(frame):
    """The whole point: an aligned car sits in the box and heads along ego +X+yaw."""
    _, cam = frame
    box = cam.boxes[0]
    rot_cam_box = box.rotation.rotation_matrix
    pos_cam_box = np.asarray(box.position)

    quat, trans, scale = fake_sam3d_output(
        rot_cam_box, pos_cam_box, yaw_error=np.deg2rad(11.0), scale_error=0.45
    )
    alignment = align_to_box(
        canonical_car(),
        sam3d_rotation=quat,
        sam3d_translation=trans,
        sam3d_scale=scale,
        box_rotation=rot_cam_box,
        box_position=pos_cam_box,
        box_size=np.asarray(box.size),
        percentile=0.0,
    )
    in_camera = compose_alignment(alignment, rot_cam_box, pos_cam_box, "camera")
    in_ego = compose_alignment(in_camera, cam.rot_ego_cam, cam.trans_ego_cam, "base_link")

    aligned = canonical_car() @ in_ego.linear.T + in_ego.translation
    lower, upper, _ = robust_extent(aligned, percentile=0.0)
    assert (lower + upper) / 2 == pytest.approx(CAR_IN_EGO, abs=1e-9)

    forward = in_ego.linear @ np.array([0.0, 1.0, 0.0])
    forward /= np.linalg.norm(forward)
    assert np.arctan2(forward[1], forward[0]) == pytest.approx(CAR_YAW, abs=1e-9)
    assert forward[2] == pytest.approx(0.0, abs=1e-9)

    # measured in the box's own frame the reconstruction is exactly the annotation
    local = (aligned - CAR_IN_EGO) @ rotz(CAR_YAW)
    width, length, height = BOX_SIZE_WLH
    assert local.max(axis=0) == pytest.approx([length / 2, width / 2, height / 2], rel=1e-9)

    report = alignment.report
    assert report["yaw_correction_deg"] == pytest.approx(-11.0, abs=1e-6)
    assert np.mean(report["scale_ratio_vs_sam3d"]) == pytest.approx(1 / 0.45, rel=1e-9)
