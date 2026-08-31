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
    "sharpness",
    "view_score",
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
    #: Angle between the line of sight and the object's forward axis, 0-90 deg.
    #: 0 is nose- or tail-on, 90 is broadside, 45 is the three-quarter view.
    aspect_deg: float = 45.0

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
            "aspect_deg": self.aspect_deg,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ObjectView":
        return cls(
            **{
                field: payload[field]
                for field in cls.__dataclass_fields__
                if field in payload
            }
        )


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
                    aspect_deg=_aspect_angle(box),
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


def _aspect_angle(box) -> float:
    """Angle between the line of sight and the object's forward axis, in degrees.

    Measured in the horizontal plane and folded into 0-90: a vehicle looks the
    same to this whether it is coming or going.
    """
    forward = np.asarray(box.rotation.rotation_matrix, float)[:, 0]
    towards = np.asarray(box.position, float)
    # The camera frame is OpenCV: +X right, +Y down, +Z forward, so the ground
    # plane is XZ and "down" is the axis to drop.
    a = np.array([forward[0], forward[2]])
    b = np.array([towards[0], towards[2]])
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return 45.0
    cosine = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def view_score(
    view: ObjectView,
    *,
    reference_area: float,
    aspect_weight: float = 0.35,
    lidar_weight: float = 0.15,
) -> float:
    """How promising a frame is as the single input to a reconstruction.

    Three things decide it, and size alone is none of them:

    * **size** -- pixels on the object, relative to the best frame of that
      instance. Taken as a square root, because the difference between 20% and
      40% of the best matters far more than between 80% and 100%.
    * **aspect** -- a three-quarter view shows two faces at once, which is what
      pins an object's length *and* width. Nose-on shows one, and SAM 3D has to
      guess the rest of the vehicle.
    * **lidar support** -- returns on the object are what the geometry refit and
      the reflectance fit are later built from.

    Returns a number in roughly 0-1; only the ordering is meaningful.
    """
    size = np.sqrt(np.clip(view.area_px / max(reference_area, 1.0), 0.0, 1.0))
    # Peaks at 45 deg, and never falls to zero: a broadside view is worse than a
    # three-quarter one, not useless.
    aspect = 0.5 + 0.5 * float(np.sin(np.radians(2 * view.aspect_deg)))
    lidar = np.log1p(view.num_lidar_pts) / np.log1p(5000.0)
    base = 1.0 - aspect_weight - lidar_weight
    return float(base * size + aspect_weight * aspect + lidar_weight * min(lidar, 1.0))


def sharpness(t4, view: ObjectView) -> float:
    """Focus of the object's own pixels, as the variance of their Laplacian.

    A frame can be large, well angled and well covered by lidar and still be a
    poor reconstruction input because the vehicle was moving across it. This is
    the one term that needs the image decoded, so it is measured on a shortlist
    rather than on every frame of the scan.
    """
    import cv2

    from .dataset import load_camera_frame

    frame = load_camera_frame(t4, sample_token=view.sample_token, channel=view.camera)
    box = next((b for b in frame.boxes if b.uuid == view.instance_token), None)
    if box is None:
        return 0.0
    uv = projected_corners(frame, box)
    if uv is None:
        return 0.0
    width, height = frame.size
    x0, x1 = np.clip([uv[:, 0].min(), uv[:, 0].max()], 0, width - 1).astype(int)
    y0, y1 = np.clip([uv[:, 1].min(), uv[:, 1].max()], 0, height - 1).astype(int)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    crop = cv2.cvtColor(frame.image[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def best_per_instance(
    views: Sequence[ObjectView],
    *,
    t4=None,
    shortlist: int = 5,
    sharpness_weight: float = 0.3,
) -> list[ObjectView]:
    """The frame to reconstruct each instance from, best first.

    Ranked by :func:`view_score`, then -- when ``t4`` is given -- the top
    ``shortlist`` frames of each instance have their focus measured and the
    ranking is redone. Decoding five images per object is affordable; decoding
    every frame of the scan is not.
    """
    by_instance: dict[str, list[ObjectView]] = {}
    for view in views:
        by_instance.setdefault(view.instance_token, []).append(view)

    chosen = []
    for candidates in by_instance.values():
        reference = max(v.area_px for v in candidates)
        ranked = sorted(
            candidates, key=lambda v: -view_score(v, reference_area=reference)
        )
        best = ranked[0]
        if t4 is not None and shortlist > 1:
            top = ranked[:shortlist]
            focus = [sharpness(t4, v) for v in top]
            sharpest = max(focus) or 1.0
            best = max(
                zip(top, focus),
                key=lambda pair: view_score(pair[0], reference_area=reference)
                + sharpness_weight * (pair[1] / sharpest),
            )[0]
        chosen.append((best, view_score(best, reference_area=reference)))
    return [view for view, _ in sorted(chosen, key=lambda pair: -pair[1])]
