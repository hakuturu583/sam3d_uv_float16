"""Load T4 dataset camera frames and object masks with ``t4-devkit`` v0.8.0.

Only the pieces SAM 3D needs are pulled out: the RGB image, the camera
intrinsics/distortion, the 3D boxes expressed **in the camera sensor frame**
(OpenCV convention) and a per-object mask to drive the reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "CameraFrame",
    "box_mask",
    "load_t4",
    "load_camera_frame",
    "project_points",
    "select_boxes",
]


def load_t4(data_root, revision: str | None = None, verbose: bool = True):
    """Open a T4 dataset directory.

    Uses ``T4Devkit`` (v0.8.0) and falls back to the deprecated ``Tier4`` alias
    for older releases.
    """
    try:
        from t4_devkit import T4Devkit
    except ImportError:  # t4-devkit < 0.8
        from t4_devkit import Tier4 as T4Devkit  # type: ignore[attr-defined]
    return T4Devkit(data_root, revision=revision, verbose=verbose)


@dataclass
class CameraFrame:
    """One camera image plus everything needed to place objects around it.

    Attributes:
        sample_token: Token of the owning ``sample``.
        sample_data_token: Token of the image ``sample_data``.
        channel: Sensor channel, e.g. ``"CAM_FRONT"``.
        image: ``(H, W, 3)`` uint8 RGB image.
        intrinsic: ``(3, 3)`` camera matrix.
        distortion: ``(n,)`` OpenCV distortion coefficients, or ``None``.
        boxes: ``Box3D`` list **in the camera sensor frame** (OpenCV axes).
        rot_ego_cam / trans_ego_cam: camera -> ``base_link`` rigid transform.
        rot_map_ego / trans_map_ego: ``base_link`` -> ``map`` rigid transform.
    """

    sample_token: str
    sample_data_token: str
    channel: str
    image: np.ndarray
    intrinsic: np.ndarray
    distortion: np.ndarray | None
    boxes: list[Any]
    rot_ego_cam: np.ndarray
    trans_ego_cam: np.ndarray
    rot_map_ego: np.ndarray
    trans_map_ego: np.ndarray

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.image.shape[1], self.image.shape[0]


def load_camera_frame(
    t4,
    *,
    sample_token: str | None = None,
    sample_index: int = 0,
    channel: str = "CAM_FRONT",
    visibility: str | None = None,
) -> CameraFrame:
    """Read one camera image and its 3D boxes from an open T4 dataset.

    Args:
        t4: An open ``T4Devkit``.
        sample_token: Sample to read. Defaults to ``t4.sample[sample_index]``.
        sample_index: Index into ``t4.sample`` when ``sample_token`` is ``None``.
        channel: Camera channel name.
        visibility: Optional ``VisibilityLevel`` name (``"none"``, ``"partial"``,
            ``"most"``, ``"full"``) used to drop boxes outside the image.
    """
    from t4_devkit.schema import VisibilityLevel

    sample = t4.get("sample", sample_token) if sample_token else t4.sample[sample_index]
    if channel not in sample.data:
        raise KeyError(f"channel {channel!r} not in sample; available: {sorted(sample.data)}")
    sd_token = sample.data[channel]

    level = VisibilityLevel.NONE if visibility is None else VisibilityLevel(visibility)
    data_path, boxes, intrinsic = t4.get_sample_data(
        sd_token,
        as_3d=True,
        as_sensor_coord=True,
        visibility=level,
    )
    if intrinsic is None:
        raise ValueError(f"{channel!r} is not a camera channel")

    sd_record = t4.get("sample_data", sd_token)
    cs_record = t4.get("calibrated_sensor", sd_record.calibrated_sensor_token)
    pose_record = t4.get("ego_pose", sd_record.ego_pose_token)

    distortion = np.asarray(cs_record.camera_distortion, dtype=np.float64).reshape(-1)
    if distortion.size == 0 or not np.any(distortion):
        distortion = None

    image = np.asarray(Image.open(data_path).convert("RGB"), dtype=np.uint8)

    return CameraFrame(
        sample_token=sample.token,
        sample_data_token=sd_token,
        channel=channel,
        image=image,
        intrinsic=np.asarray(intrinsic, dtype=np.float64).reshape(3, 3),
        distortion=distortion,
        boxes=list(boxes),
        rot_ego_cam=np.asarray(cs_record.rotation.rotation_matrix, dtype=np.float64),
        trans_ego_cam=np.asarray(cs_record.translation, dtype=np.float64).reshape(3),
        rot_map_ego=np.asarray(pose_record.rotation.rotation_matrix, dtype=np.float64),
        trans_map_ego=np.asarray(pose_record.translation, dtype=np.float64).reshape(3),
    )


def select_boxes(
    frame: CameraFrame,
    *,
    categories: Sequence[str] | None = None,
    min_area_px: float = 0.0,
    max_distance: float | None = None,
    min_lidar_points: int | None = None,
) -> list[Any]:
    """Filter a frame's boxes down to the ones worth reconstructing.

    Args:
        frame: A :class:`CameraFrame`.
        categories: Keep only these category names (substring match, so
            ``"car"`` also matches ``"vehicle.car"``). ``None`` keeps all.
        min_area_px: Drop boxes whose projected 2D footprint is smaller than this.
        max_distance: Drop boxes farther than this many metres from the camera.
        min_lidar_points: Drop boxes with fewer annotated LiDAR points.
    """
    kept = []
    for box in frame.boxes:
        name = box.semantic_label.name
        if categories is not None and not any(c in name for c in categories):
            continue
        if max_distance is not None and float(np.linalg.norm(box.position)) > max_distance:
            continue
        if min_lidar_points is not None and (box.num_points or 0) < min_lidar_points:
            continue
        if min_area_px > 0.0:
            uv = project_points(box.corners(), frame.intrinsic, frame.distortion)
            if uv is None:
                continue
            area = float(np.prod(uv.max(axis=0) - uv.min(axis=0)))
            if area < min_area_px:
                continue
        kept.append(box)
    return kept


def project_points(points, intrinsic, distortion=None) -> np.ndarray | None:
    """Project camera-frame points to pixels, returning ``None`` if all are behind.

    Points with non-positive depth are dropped. ``t4_devkit.common.geometry.view_points``
    is deliberately not used here: on the distortion path it expects normalised
    image coordinates rather than 3D points, which is easy to get wrong.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[pts[:, 2] > 1e-3]
    if pts.shape[0] == 0:
        return None

    normalized = pts[:, :2] / pts[:, 2:3]
    if distortion is not None and np.any(distortion):
        try:
            import cv2

            uv, _ = cv2.projectPoints(
                pts.reshape(-1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                np.asarray(intrinsic, dtype=np.float64),
                np.asarray(distortion, dtype=np.float64).reshape(1, -1),
            )
            return uv.reshape(-1, 2)
        except ImportError:
            pass  # fall through to the pinhole projection

    k = np.asarray(intrinsic, dtype=np.float64)
    return np.stack(
        (
            k[0, 0] * normalized[:, 0] + k[0, 1] * normalized[:, 1] + k[0, 2],
            k[1, 1] * normalized[:, 1] + k[1, 2],
        ),
        axis=1,
    )


def box_mask(
    t4,
    frame: CameraFrame,
    box,
    *,
    source: str = "auto",
    dilate: int = 0,
) -> np.ndarray | None:
    """Return a ``(H, W)`` boolean mask for one box, to feed SAM 3D.

    Args:
        t4: The open ``T4Devkit``.
        frame: The :class:`CameraFrame` the box came from.
        box: A ``Box3D`` from ``frame.boxes``.
        source: ``"ann"`` uses the annotated instance mask from ``object_ann``
            (accurate, but only present in datasets with 2D annotations);
            ``"hull"`` fills the convex hull of the projected 3D box corners
            (always available, but includes background around the object);
            ``"auto"`` prefers the annotation and falls back to the hull.
        dilate: Grow the mask by this many pixels.

    Returns:
        The mask, or ``None`` if the box projects entirely behind the camera.
    """
    if source not in ("auto", "ann", "hull"):
        raise ValueError(f"source must be 'auto', 'ann' or 'hull', got {source!r}")

    mask = None
    if source in ("auto", "ann"):
        mask = _annotated_mask(t4, frame, box)
    if mask is None and source in ("auto", "hull"):
        mask = _hull_mask(frame, box)
    if mask is None:
        return None

    if dilate > 0:
        mask = _dilate(mask, dilate)
    return mask


def _annotated_mask(t4, frame: CameraFrame, box) -> np.ndarray | None:
    """Decode the ``object_ann`` RLE mask for this instance, if the dataset has one."""
    for ann in getattr(t4, "object_ann", []):
        if ann.sample_data_token != frame.sample_data_token:
            continue
        if ann.instance_token != box.uuid:
            continue
        decoded = np.asarray(ann.mask.decode()).astype(bool)
        if decoded.shape != frame.image.shape[:2]:
            return None
        return decoded
    return None


def _hull_mask(frame: CameraFrame, box) -> np.ndarray | None:
    """Fill the convex hull of the projected 3D box corners."""
    uv = project_points(box.corners(), frame.intrinsic, frame.distortion)
    if uv is None or uv.shape[0] < 3:
        return None

    width, height = frame.size
    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(filled, hull.astype(np.int32), 1)
        return filled.astype(bool)

    # Pure numpy fallback: the axis-aligned bounding box of the projection.
    mask = np.zeros((height, width), dtype=bool)
    x0, y0 = np.floor(uv.min(axis=0)).astype(int)
    x1, y1 = np.ceil(uv.max(axis=0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, width), min(y1, height)
    if x1 <= x0 or y1 <= y0:
        return None
    mask[y0:y1, x0:x1] = True
    return mask


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    try:
        import cv2

        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    except ImportError:
        from scipy.ndimage import binary_dilation

        return binary_dilation(mask, iterations=radius)
