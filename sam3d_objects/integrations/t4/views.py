"""Pick the camera frames an object is worth reading out of.

Every stage of the T4 pipeline asks the same question in a slightly different
way -- which frames show this object cleanly? -- and got its own answer while
the pipeline grew: one copy in the mask tool, one in the lidar module, one in a
throwaway script. They drifted. This module is the single answer.

"Cleanly" means two things, and both matter:

* **unoccluded** -- the annotator's ``visibility="full"``, i.e. nothing in front
  of the object;
* **whole** -- all eight corners of its 3D box project inside the frame, with a
  margin, so the reconstruction is not fed a vehicle cut in half by the edge.

The scan reads boxes and calibration only, never the pixels: decoding every
frame of every camera to throw the image away costs more than the rest of the
search put together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "CAMERAS",
    "ObjectView",
    "best_per_instance",
    "fully_inside",
    "projected_corners",
    "scan_views",
]

#: Cameras worth searching for a clean view of an object.
CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT_WIDE",
    "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT_RIGHT_WIDE",
    "CAM_BACK_LEFT_WIDE",
    "CAM_BACK_RIGHT_WIDE",
)


@dataclass(frozen=True)
class ObjectView:
    """One object seen cleanly in one camera frame."""

    sample_index: int
    sample_token: str
    camera: str
    instance_token: str
    category: str
    area_px: float
    distance_m: float
    num_lidar_pts: int

    def to_dict(self) -> dict:
        return {
            "sample_index": self.sample_index,
            "sample_token": self.sample_token,
            "camera": self.camera,
            "instance_token": self.instance_token,
            "category": self.category,
            "area_px": self.area_px,
            "distance_m": self.distance_m,
            "num_lidar_pts": self.num_lidar_pts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ObjectView":
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__})


def projected_corners(frame, box) -> np.ndarray | None:
    """The box's eight corners in pixels, or ``None`` if any sits behind the camera.

    Args:
        frame: A :class:`~sam3d_objects.integrations.t4.dataset.CameraFrame` or
            :class:`~sam3d_objects.integrations.t4.dataset.CameraGeometry`.
        box: A ``Box3D`` in that frame's sensor coordinates.
    """
    from .dataset import project_points

    corners = np.asarray(box.corners()).reshape(-1, 3)
    if corners.shape[0] != 8 or (corners[:, 2] <= 0.5).any():
        return None
    uv = project_points(corners, frame.intrinsic, frame.distortion)
    return uv if uv is not None and uv.shape[0] == 8 else None


def fully_inside(uv: np.ndarray, size: tuple[int, int], margin: float) -> bool:
    """Whether every projected corner clears the frame border by ``margin`` pixels."""
    width, height = size
    return bool(
        uv[:, 0].min() >= margin
        and uv[:, 0].max() <= width - margin
        and uv[:, 1].min() >= margin
        and uv[:, 1].max() <= height - margin
    )


def scan_views(
    t4,
    *,
    cameras: Sequence[str] = CAMERAS,
    categories: Iterable[str] | None = None,
    instance_token: str | None = None,
    visibility: str | None = "full",
    margin: int = 25,
    require_whole: bool = True,
    min_lidar_pts: int = 0,
    max_distance: float | None = None,
    best_per_sample: bool = True,
    sample_stride: int = 1,
    progress=None,
) -> list[ObjectView]:
    """Every clean view of every object the filters keep.

    Args:
        t4: An open ``T4Devkit``.
        cameras: Camera channels to search.
        categories: Keep only these category names (substring match).
        instance_token: Keep only this instance.
        visibility: Annotated visibility floor; ``"full"`` is "no occlusion".
        margin: Border, in pixels, the projected box must keep clear.
        require_whole: Drop boxes whose projection leaves the frame.
        min_lidar_pts: Drop views whose annotation counts fewer lidar points.
        max_distance: Drop objects farther than this, in metres.
        best_per_sample: Keep one view per (sample, instance) -- the camera that
            sees the object largest. A second camera of the same instant adds a
            duplicate of the same lidar sweep, not new information.
        sample_stride: Look at every Nth sample; 1 reads them all.
        progress: Optional callable taking ``(sample_index, n_samples, n_views)``.

    Returns:
        Views ordered by sample index, then camera.
    """
    from .dataset import load_camera_geometry

    wanted = tuple(categories) if categories else None
    best: dict[tuple[int, str], ObjectView] = {}
    found: list[ObjectView] = []
    n_samples = len(t4.sample)

    for sample_index in range(0, n_samples, sample_stride):
        for camera in cameras:
            try:
                frame = load_camera_geometry(
                    t4, sample_index=sample_index, channel=camera, visibility=visibility
                )
            except Exception:
                continue
            for box in frame.boxes:
                name = box.semantic_label.name
                if instance_token is not None and box.uuid != instance_token:
                    continue
                if wanted is not None and not any(c in name for c in wanted):
                    continue
                if (box.num_points or 0) < min_lidar_pts:
                    continue
                distance = float(np.linalg.norm(box.position))
                if max_distance is not None and distance > max_distance:
                    continue
                uv = projected_corners(frame, box)
                if uv is None or (require_whole and not fully_inside(uv, frame.size, margin)):
                    continue

                view = ObjectView(
                    sample_index=sample_index,
                    sample_token=frame.sample_token,
                    camera=camera,
                    instance_token=box.uuid,
                    category=name,
                    area_px=float(
                        (uv[:, 0].max() - uv[:, 0].min()) * (uv[:, 1].max() - uv[:, 1].min())
                    ),
                    distance_m=distance,
                    num_lidar_pts=int(box.num_points or 0),
                )
                if not best_per_sample:
                    found.append(view)
                    continue
                key = (sample_index, box.uuid)
                if key not in best or view.area_px > best[key].area_px:
                    best[key] = view
        if progress is not None:
            progress(sample_index, n_samples, len(best) if best_per_sample else len(found))

    views = list(best.values()) if best_per_sample else found
    return sorted(views, key=lambda v: (v.sample_index, v.camera))


def best_per_instance(views: Sequence[ObjectView]) -> list[ObjectView]:
    """The single best view of each instance -- the one that sees it largest.

    This is what a reconstruction wants: one frame, as much of the object in it
    as the sequence ever offers.
    """
    best: dict[str, ObjectView] = {}
    for view in views:
        current = best.get(view.instance_token)
        if current is None or view.area_px > current.area_px:
            best[view.instance_token] = view
    return sorted(best.values(), key=lambda v: -v.area_px)
