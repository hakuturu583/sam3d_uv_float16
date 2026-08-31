"""Give a reconstructed object the lidar attributes ``splatad_kernel`` renders.

SAM 3D returns colour, not reflectivity, so a splat cloud cannot be fed to a
lidar rasterizer as it stands. This module fills that gap from the dataset's own
returns: every unoccluded view of the object contributes the lidar points that
land inside its SAM 3 mask, and those measurements supervise a per-Gaussian
intensity and ray-drop feature while the geometry stays exactly as reconstructed.

``splatad_kernel`` alpha-composites the feature channels and returns them
directly -- there is no decoder network behind them -- so the stored value has to
be an *intrinsic*, view-independent reflectivity. Raw lidar intensity is not:
it carries the beam's incidence angle on the surface and, depending on the
sensor's firmware, its range. Those are removed here before fitting, which is
the linearisation to an "equivalent Lambertian reflectance" that the radiometric
calibration literature describes; :func:`fit_reflectance` measures how much of
each effect the data actually shows rather than assuming the textbook form.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "ViewReturns",
    "INTENSITY_SCALE",
    "ObjectReturns",
    "ReflectanceModel",
    "collect_returns",
    "fit_reflectance",
    "lidar_frame",
    "GeometryFit",
    "load_returns",
    "refine_to_lidar",
    "save_returns",
    "unoccluded_views",
]

#: Lidar channel holding the merged cloud.
LIDAR_CHANNEL = "LIDAR_CONCAT"


@dataclass
class ObjectReturns:
    """Lidar returns from one object, pooled over views.

    Attributes:
        xyz: ``(N, 3)`` hit positions in the object's box frame.
        intensity: ``(N,)`` raw sensor intensity, 0-255.
        distance: ``(N,)`` range from the sensor to the hit, in metres.
        cos_incidence: ``(N,)`` cosine of the angle between the beam and the
            surface normal, from the box face the hit sits on.
        colour: ``(N, 3)`` RGB the camera saw at that hit, 0-1.
        view_index: ``(N,)`` which view each return came from.
    """

    xyz: np.ndarray
    intensity: np.ndarray
    distance: np.ndarray
    cos_incidence: np.ndarray
    colour: np.ndarray
    view_index: np.ndarray

    def __len__(self) -> int:
        return len(self.xyz)

    def concat(self, other: "ObjectReturns") -> "ObjectReturns":
        return ObjectReturns(
            *(
                np.concatenate([getattr(self, f), getattr(other, f)])
                for f in ("xyz", "intensity", "distance", "cos_incidence", "colour", "view_index")
            )
        )


@dataclass
class ViewReturns:
    """One view's returns, with what the rasterizer needs to reproduce that view.

    ``returns`` is in the object's box frame -- the frame the splats live in --
    while ``hits_sensor`` keeps the same points where the beams were actually
    measured, which is the only place azimuth, elevation and range mean anything.
    """

    returns: "ObjectReturns"
    hits_sensor: np.ndarray  # (N, 3)
    viewmat: np.ndarray  # (4, 4) box frame -> sensor


def unoccluded_views(t4, instance_token: str, **kwargs):
    """Every sample where the object is unoccluded and entirely inside a camera.

    A thin wrapper over :func:`~sam3d_objects.integrations.t4.views.scan_views`,
    which every stage of the pipeline shares; the only thing lidar work adds is
    that a view with no lidar points on the object is of no use to it.
    """
    from .views import scan_views

    kwargs.setdefault("min_lidar_pts", 1)
    return scan_views(t4, instance_token=instance_token, **kwargs)


def lidar_frame(t4, sample_token: str):
    """Read one lidar sweep: ``(points, intensity, boxes, sensor->ego, ego->map)``.

    Points come back in the sensor's own frame, which is where the ranges and
    incidence angles have to be measured.
    """
    sample = t4.get("sample", sample_token)
    sd_token = sample.data[LIDAR_CHANNEL]
    path, boxes, _ = t4.get_sample_data(sd_token, as_3d=True, as_sensor_coord=True)

    raw = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
    points, intensity = raw[:, :3].astype(np.float64), raw[:, 3].astype(np.float64)

    sd_record = t4.get("sample_data", sd_token)
    cs = t4.get("calibrated_sensor", sd_record.calibrated_sensor_token)
    pose = t4.get("ego_pose", sd_record.ego_pose_token)
    return (
        points,
        intensity,
        list(boxes),
        (np.asarray(cs.rotation.rotation_matrix, float), np.asarray(cs.translation, float).reshape(3)),
        (np.asarray(pose.rotation.rotation_matrix, float), np.asarray(pose.translation, float).reshape(3)),
    )


def _box_frame_points(points: np.ndarray, box) -> np.ndarray:
    """Sensor-frame points expressed in the box's own frame (+X forward, +Z up)."""
    rot = np.asarray(box.rotation.rotation_matrix, float)
    return (points - np.asarray(box.position, float)) @ rot


def _face_normals(local: np.ndarray, size_wlh) -> np.ndarray:
    """Unit normal of the box face each hit is closest to, in the box frame.

    A splat cloud has per-Gaussian normals, but they are only as trustworthy as
    the reconstruction; the annotated box is the one piece of geometry that is
    ground truth here, and for vehicles its faces carry the incidence angle well
    enough to divide it out.
    """
    width, length, height = (float(v) for v in size_wlh)
    half = np.array([length, width, height]) / 2.0
    # Distance to each of the six faces, as a fraction of the half-extent, so the
    # nearest face is the one the surface most plausibly belongs to.
    slack = np.abs(np.abs(local) / half) - 1.0
    axis = np.argmax(slack, axis=1)
    normal = np.zeros_like(local)
    normal[np.arange(len(local)), axis] = np.sign(local[np.arange(len(local)), axis])
    return normal


@dataclass
class ReflectanceModel:
    """Maps a Gaussian's colour to the reflectivity a lidar would measure.

    ``intensity = gain * reflectance(colour) * cos(incidence)^cos_power /
    distance^range_power``, with the geometric terms divided out of the
    measurements before the colour term is fitted. ``range_power`` is 0 for a
    sensor that already reports range-compensated reflectivity, which is what the
    fit decides from the data.
    """

    weights: np.ndarray  # (4,) [R, G, B, bias] -> reflectance
    gain: float
    cos_power: float
    range_power: float
    r2: float
    n_points: int

    def predict(self, colour: np.ndarray) -> np.ndarray:
        """Per-Gaussian reflectivity, on the same 0-255 scale as the sensor."""
        colour = np.asarray(colour, float).reshape(-1, 3)
        design = np.concatenate([colour, np.ones((len(colour), 1))], axis=1)
        return np.clip(design @ self.weights, 0.0, 255.0)

    def to_dict(self) -> dict:
        return {
            "weights_rgb_bias": [float(v) for v in self.weights],
            "gain": self.gain,
            "cos_power": self.cos_power,
            "range_power": self.range_power,
            "r2": self.r2,
            "n_points": self.n_points,
        }


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    residual = float(np.sum((y - pred) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else 0.0


def fit_reflectance(
    returns: ObjectReturns,
    *,
    cos_powers: Sequence[float] = (0.0, 0.5, 1.0),
    range_powers: Sequence[float] = (0.0, 1.0, 2.0),
    min_cos: float = 0.15,
) -> ReflectanceModel:
    """Fit colour -> reflectivity, dividing out whichever geometric terms help.

    The classical lidar radiometric model is ``I = C * rho * cos(theta) / R^2``.
    Automotive sensors usually publish an already range-compensated reflectivity,
    so the exponents are not assumed: each ``(cos_power, range_power)`` pair is
    tried, the geometric factor is divided out of the measurement, a linear
    colour model is least-squares fitted to what remains, and the combination
    that explains the raw intensities best wins.

    Args:
        returns: Pooled measurements for one object.
        cos_powers: Incidence-angle exponents to try.
        range_powers: Range exponents to try.
        min_cos: Drop grazing hits below this cosine; their correction explodes.

    Returns:
        The best :class:`ReflectanceModel`, whose ``r2`` is measured against the
        raw intensities so the variants are comparable.
    """
    keep = returns.cos_incidence > min_cos
    colour = returns.colour[keep]
    intensity = returns.intensity[keep]
    cos_incidence = returns.cos_incidence[keep]
    distance = np.maximum(returns.distance[keep], 1e-3)
    design = np.concatenate([colour, np.ones((len(colour), 1))], axis=1)

    best = None
    for cos_power in cos_powers:
        for range_power in range_powers:
            geometry = cos_incidence**cos_power / distance**range_power
            # Fit reflectance against the geometry-free part of the measurement,
            # weighted by the geometry so bright, well-conditioned hits dominate.
            target = intensity / np.maximum(geometry, 1e-9)
            weight = geometry / geometry.mean()
            lhs = design * weight[:, None]
            weights, *_ = np.linalg.lstsq(lhs, target * weight, rcond=None)
            predicted = np.clip(design @ weights, 0.0, 255.0) * geometry
            score = _r2(intensity, predicted)
            if best is None or score > best.r2:
                best = ReflectanceModel(
                    weights=weights,
                    gain=1.0,
                    cos_power=float(cos_power),
                    range_power=float(range_power),
                    r2=float(score),
                    n_points=int(keep.sum()),
                )
    return best


def collect_returns(
    t4,
    view,
    instance_token: str,
    mask_dir,
    *,
    inflate: float = 0.10,
    view_index: int = 0,
) -> ViewReturns | None:
    """Lidar returns from one view that land on the object.

    A point qualifies when it is inside the annotated 3D box *and* projects into
    the SAM 3 mask for that image. The box alone keeps the road under the vehicle
    and whatever shares its footprint; the mask is what makes the set the object's
    own surface.

    Args:
        t4: An open ``T4Devkit``.
        view: One :class:`~sam3d_objects.integrations.t4.views.ObjectView`.
        instance_token: The object to keep.
        mask_dir: Where ``tools/t4_sam3_masks.py`` wrote its masks.
        inflate: Grow the box by this fraction before testing containment, so
            hits on the skin of a slightly tight annotation survive.
        view_index: Stamped onto every return, for per-view bookkeeping.
    """
    from PIL import Image

    from .dataset import load_camera_frame, mask_key, project_points

    frame = load_camera_frame(t4, sample_token=view.sample_token, channel=view.camera)
    box_cam = next((b for b in frame.boxes if b.uuid == instance_token), None)
    if box_cam is None:
        return None

    mask_path = os.path.join(str(mask_dir), f"{mask_key(frame, box_cam)}.png")
    if not os.path.exists(mask_path):
        return None
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127

    points, intensity, boxes, (rot_ego_lidar, trans_ego_lidar), (rot_map_ego_l, trans_map_ego_l) = (
        lidar_frame(t4, view.sample_token)
    )
    box_lidar = next((b for b in boxes if b.uuid == instance_token), None)
    if box_lidar is None:
        return None

    local = _box_frame_points(points, box_lidar)
    width, length, height = (float(v) for v in box_lidar.size)
    half = np.array([length, width, height]) / 2.0 * (1.0 + inflate)
    inside = np.all(np.abs(local) <= half, axis=1)
    if not inside.any():
        return None

    local, hits, intensity = local[inside], points[inside], intensity[inside]

    # Lidar -> map -> the camera's ego pose -> camera, so the two sensors' own
    # timestamps and calibrations are each honoured instead of assumed equal.
    ego = hits @ rot_ego_lidar.T + trans_ego_lidar
    world = ego @ rot_map_ego_l.T + trans_map_ego_l
    ego_cam = (world - frame.trans_map_ego) @ frame.rot_map_ego
    cam = (ego_cam - frame.trans_ego_cam) @ frame.rot_ego_cam

    uv = project_points(cam, frame.intrinsic, frame.distortion)
    in_front = cam[:, 2] > 1e-3
    if uv is None or not in_front.any():
        return None

    image_h, image_w = mask.shape
    pixel = np.round(uv).astype(int)
    valid = (
        (pixel[:, 0] >= 0) & (pixel[:, 0] < image_w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < image_h)
    )
    on_object = np.zeros(len(pixel), bool)
    on_object[valid] = mask[pixel[valid, 1], pixel[valid, 0]]
    if not on_object.any():
        return None

    keep_front = np.flatnonzero(in_front)[on_object]
    local, hits, intensity = local[keep_front], hits[keep_front], intensity[keep_front]
    pixel = pixel[np.flatnonzero(in_front)][on_object]

    distance = np.linalg.norm(hits, axis=1)
    beam = hits / np.maximum(distance, 1e-9)[:, None]
    # The beam direction in the box frame, against the face normal there.
    beam_local = beam @ np.asarray(box_lidar.rotation.rotation_matrix, float)
    normal = _face_normals(local, box_lidar.size)
    cos_incidence = np.abs(np.sum(beam_local * normal, axis=1))

    colour = frame.image[pixel[:, 1], pixel[:, 0]].astype(np.float64) / 255.0

    # The box frame is the splats' world, so world -> sensor is the box's own pose.
    viewmat = np.eye(4)
    viewmat[:3, :3] = np.asarray(box_lidar.rotation.rotation_matrix, float)
    viewmat[:3, 3] = np.asarray(box_lidar.position, float)
    return ViewReturns(
        returns=ObjectReturns(
            xyz=local,
            intensity=intensity,
            distance=distance,
            cos_incidence=cos_incidence,
            colour=colour,
            view_index=np.full(len(local), view_index, dtype=np.int64),
        ),
        hits_sensor=hits,
        viewmat=viewmat,
    )


# --- training ---------------------------------------------------------------
#
# What the optimiser is allowed to touch: the three per-Gaussian channels a
# lidar rasterizer reads. Geometry -- means, quaternions, scales -- is frozen,
# because it came from SAM 3D and the lidar returns are far too sparse to
# improve it without wrecking it.

#: Channel layout of the ``lidar_features`` tensor handed to ``splatad_kernel``.
FEATURE_INTENSITY, FEATURE_RAYDROP = 0, 1

#: Sensor intensity is reported 0-255; features are carried normalised.
INTENSITY_SCALE = 255.0


@dataclass
class BeamGrid:
    """One view's measured returns, on the spherical grid the kernel rasterizes.

    The rasterizer wants a regular (elevation x azimuth) grid, while a merged
    lidar sweep is a bag of points with no ring index left. Only the cells that
    received a return are kept -- a few thousand out of a full ring -- and the
    dense tensors the kernel wants are built per step by :meth:`dense`.

    Azimuth spans the whole 360 deg ring. ``splatad_kernel`` bins Gaussians into
    a tile grid that always covers the full ring; its wrap arithmetic assumes it.
    A narrow image is expressible as a *sector* via ``tile_col_offset``, but that
    path is forward-only -- ``rasterize_to_points`` refuses a backward pass with a
    non-zero offset -- so training renders the whole ring and masks the loss down
    to the measured cells. Rendering a sector is left to inference.
    """

    rows: np.ndarray  # (M,) elevation cell of each measured beam
    cols: np.ndarray  # (M,) azimuth cell, indexed from -180 deg
    ranges: np.ndarray  # (M,) metres
    intensity: np.ndarray  # (M,) normalised reflectance
    viewmat: np.ndarray  # (4, 4) box frame -> sensor
    height: int
    width: int
    azimuth_resolution: float
    elevation_resolution: float
    min_elevation: float
    max_elevation: float

    @property
    def n_measured(self) -> int:
        return len(self.rows)

    def dense(self, device):
        """``(raster_pts, measured, target_range, target_intensity)`` on ``device``.

        Cells with no return get range 0, which the kernel reads as "no beam
        here" and skips -- so the empty part of the ring costs almost nothing.
        """
        import torch

        rows = torch.as_tensor(self.rows, dtype=torch.long, device=device)
        cols = torch.as_tensor(self.cols, dtype=torch.long, device=device)
        shape = (self.height, self.width)

        raster = torch.zeros((1, *shape, 4), dtype=torch.float32, device=device)
        azimuth = -180.0 + (torch.arange(self.width, device=device) + 0.5) * self.azimuth_resolution
        elevation = (
            self.min_elevation
            + (torch.arange(self.height, device=device) + 0.5) * self.elevation_resolution
        )
        raster[0, ..., 0] = azimuth
        raster[0, ..., 1] = elevation[:, None]

        measured = torch.zeros(shape, dtype=torch.bool, device=device)
        target_range = torch.zeros(shape, dtype=torch.float32, device=device)
        target_intensity = torch.zeros(shape, dtype=torch.float32, device=device)
        measured[rows, cols] = True
        target_range[rows, cols] = torch.as_tensor(self.ranges, dtype=torch.float32, device=device)
        target_intensity[rows, cols] = torch.as_tensor(
            self.intensity, dtype=torch.float32, device=device
        )
        raster[0, ..., 2] = target_range
        return raster, measured, target_range, target_intensity


def beam_grid(
    returns: ObjectReturns,
    hits_sensor: np.ndarray,
    viewmat: np.ndarray,
    reflectance: np.ndarray,
    *,
    azimuth_resolution: float = 0.12,
    elevation_resolution: float = 0.20,
    max_rows: int = 384,
    tile_width: int = 8,
) -> BeamGrid | None:
    """Bin one view's returns onto the sensor's spherical grid."""
    distance = np.linalg.norm(hits_sensor, axis=1)
    azimuth = np.degrees(np.arctan2(hits_sensor[:, 1], hits_sensor[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(hits_sensor[:, 2] / np.maximum(distance, 1e-9), -1, 1)))

    el_res = elevation_resolution
    min_el, max_el = elevation.min() - el_res, elevation.max() + el_res
    height = int(np.ceil((max_el - min_el) / el_res))
    if height < 2:
        return None
    # Coarsen rather than refuse when an object fills a lot of the field of view.
    if height > max_rows:
        el_res *= height / max_rows
        height = max_rows
    max_el = min_el + height * el_res
    if not (-85.0 <= min_el and max_el <= 85.0):
        return None

    # A whole ring of columns, quantised so the tile grid divides it exactly.
    width = int(np.ceil(360.0 / azimuth_resolution / tile_width)) * tile_width
    az_res = 360.0 / width

    col = np.clip(((azimuth + 180.0) / az_res).astype(int), 0, width - 1)
    row = np.clip(((elevation - min_el) / el_res).astype(int), 0, height - 1)

    # Nearest return wins the cell: that is what a first-return sensor reports.
    order = np.argsort(-distance)
    flat = row[order] * width + col[order]
    unique_flat, first = np.unique(flat[::-1], return_index=True)
    keep = order[::-1][first]
    if len(keep) == 0:
        return None
    return BeamGrid(
        rows=row[keep],
        cols=col[keep],
        ranges=distance[keep].astype(np.float32),
        intensity=np.asarray(reflectance, np.float32)[keep],
        viewmat=viewmat,
        height=height,
        width=width,
        azimuth_resolution=float(az_res),
        elevation_resolution=float(el_res),
        min_elevation=float(min_el),
        max_elevation=float(max_el),
    )


def train_lidar_features(
    geometry,
    grids: Sequence[BeamGrid],
    initial_intensity: np.ndarray,
    *,
    camera_opacity: np.ndarray | None = None,
    haze_opacity: float = 0.05,
    oversize_ratio: float = 20.0,
    epochs: int = 6,
    learning_rate: float = 0.02,
    opacity_init: float = 0.99,
    raydrop_init: float = -4.0,
    weight_range: float = 0.05,
    weight_raydrop: float = 0.1,
    weight_alpha: float = 0.05,
    weight_line_of_sight: float = 0.2,
    line_of_sight_margin: float = 0.2,
    tile_width: int = 8,
    tile_height: int = 8,
    device: str = "cuda",
    log=print,
):
    """Fit per-Gaussian lidar channels to the measured beams, geometry frozen.

    Args:
        geometry: ``(means, quats, scales)`` numpy arrays in the object's box frame.
        grids: Per-view :class:`BeamGrid` supervision.
        initial_intensity: ``(N,)`` normalised reflectance from the colour model,
            which is what Gaussians no beam ever hit keep.
        camera_opacity: ``(N,)`` the splats' visual opacity. A reconstruction
            carries a haze of near-transparent Gaussians around the object;
            starting those opaque to the beam builds a phantom surface in front
            of the real one, and nothing in a loss measured only where beams
            returned would ever pull them back down.
        haze_opacity: Splats below this visual opacity start at their own
            opacity rather than at ``opacity_init``.
        oversize_ratio: Splats whose largest radius exceeds this many times the
            median start transparent too. SAM 3D emits a handful of Gaussians a
            metre across on a five-metre car; spread that thin they are a faint
            wash to a camera, but to a first-return sensor they are a wall
            hanging in mid-air, and they are what puts streaks in the sweep.
        epochs: Passes over the views.
        learning_rate: Adam step.
        opacity_init: Starting opacity. A lidar beam that reaches a surface
            almost always comes back, so this starts at the top of the range and
            is only pulled down where the returns demand it.
        raydrop_init: Starting ray-drop logit; ``-4`` is a 1.8% drop rate.
        weight_range: Weight on the first-return range error.
        weight_raydrop: Weight on the ray-drop cross-entropy.
        weight_alpha: Weight pulling alpha to 1 where a beam did return.
        weight_line_of_sight: Weight on the alpha accumulated in FRONT of the
            measured return. The beam reached that surface, so the space before
            it was empty; this is what clears phantom returns.
        line_of_sight_margin: How far in front of the measurement the space is
            required to be empty, in metres.
        tile_width: Azimuth cells per tile. The rasterizer stages a tile's
            Gaussians in shared memory, so the kernel's default 32x8 asks for
            more than a consumer card grants; 8x8 fits.
        tile_height: Elevation cells per tile.
        device: Torch device.
        log: Progress sink.

    Returns:
        ``(intensity, raydrop_logit, opacity)`` as numpy arrays, plus a history
        list of per-epoch losses.
    """
    import torch
    from splatad_kernel import lidar_rasterization

    means, quats, scales = (torch.as_tensor(a, dtype=torch.float32, device=device) for a in geometry)
    n = means.shape[0]

    intensity = torch.as_tensor(initial_intensity, dtype=torch.float32, device=device).clone()
    intensity.requires_grad_(True)
    raydrop = torch.full((n,), raydrop_init, device=device, requires_grad=True)

    start_opacity = np.full(n, opacity_init, dtype=np.float64)
    radius = np.asarray(geometry[2], np.float64).max(axis=1)
    oversized = radius > oversize_ratio * np.median(radius)
    faint = (
        np.asarray(camera_opacity, np.float64) < haze_opacity
        if camera_opacity is not None
        else np.zeros(n, bool)
    )
    transparent = faint | oversized
    if transparent.any():
        source = np.asarray(camera_opacity, np.float64) if camera_opacity is not None else None
        start_opacity[transparent] = (
            np.clip(source[transparent], 1e-3, None) if source is not None else 1e-3
        )
        start_opacity[oversized] = 1e-3
        log(
            f"  {int(transparent.sum())} splats start transparent to the beam: "
            f"{int(faint.sum())} visual haze, {int(oversized.sum())} oversized "
            f"(radius above {oversize_ratio * np.median(radius) * 100:.1f} cm; "
            f"largest {radius.max() * 100:.0f} cm)"
        )
    start_opacity = np.clip(start_opacity, 1e-3, 1 - 1e-3)
    opacity_logit = torch.as_tensor(
        np.log(start_opacity / (1 - start_opacity)), dtype=torch.float32, device=device
    ).requires_grad_(True)
    optimiser = torch.optim.Adam(
        [
            {"params": [intensity], "lr": learning_rate},
            {"params": [raydrop], "lr": learning_rate},
            {"params": [opacity_logit], "lr": learning_rate * 0.5},
        ]
    )

    history = []
    for epoch in range(epochs):
        totals = np.zeros(4)
        counted = 0
        order = np.random.permutation(len(grids))
        for index in order:
            grid = grids[index]
            if grid.n_measured == 0:
                continue
            raster, measured, target_r, target_i = grid.dense(device)
            height, width = grid.height, grid.width
            viewmat = torch.as_tensor(grid.viewmat, dtype=torch.float32, device=device)[None]
            boundaries = torch.linspace(
                grid.min_elevation,
                grid.max_elevation,
                int(np.ceil(height / tile_height)) + 1,
                device=device,
            )

            features = torch.stack([intensity, raydrop], dim=-1)[None]  # (1, N, 2)
            rendered, alphas, alpha_ahead, meta = lidar_rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=torch.sigmoid(opacity_logit),
                lidar_features=features,
                velocities=None,
                viewmats=viewmat,
                raster_pts=raster,
                tile_elevation_boundaries=boundaries,
                min_azimuth=-180.0,
                max_azimuth=180.0,
                min_elevation=grid.min_elevation,
                max_elevation=grid.max_elevation,
                n_elevation_channels=height,
                azimuth_resolution=grid.azimuth_resolution,
                tile_width=tile_width,
                tile_height=tile_height,
                compute_alpha_sum_until_points=True,
                compute_alpha_sum_until_points_threshold=line_of_sight_margin,
            )

            rendered_intensity = rendered[0, ..., FEATURE_INTENSITY]
            rendered_raydrop = rendered[0, ..., FEATURE_RAYDROP]
            rendered_range = meta["median_depths"][0, ..., 0]
            alpha = alphas[0, ..., 0]

            loss_i = ((rendered_intensity - target_i)[measured] ** 2).mean()
            # Only where the model actually put a surface: a cell the splats miss
            # says nothing about range, and would otherwise drag every opacity down.
            rendered_here = measured & (rendered_range > 0)
            loss_r = (
                ((rendered_range - target_r)[rendered_here] ** 2).mean()
                if rendered_here.any()
                else torch.zeros((), device=device)
            )
            # Every measured cell returned, so its drop label is 0.
            loss_d = torch.nn.functional.binary_cross_entropy_with_logits(
                rendered_raydrop[measured], torch.zeros_like(rendered_raydrop[measured])
            )
            loss_a = ((alpha[measured] - 1.0) ** 2).mean()
            # The beam got through to the measured surface, so whatever the model
            # puts in front of it is not there. The kernel returns a SUM of
            # alphas, which is unbounded; clamped to [0, 1] it reads as "how
            # occluded was the path", which is what belongs in a loss.
            loss_front = alpha_ahead[0, ..., 0][measured].clamp(0.0, 1.0).mean()

            loss = (
                loss_i
                + weight_range * loss_r
                + weight_raydrop * loss_d
                + weight_alpha * loss_a
                + weight_line_of_sight * loss_front
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            with torch.no_grad():
                intensity.clamp_(0.0, 1.0)

            totals += [loss_i.item(), loss_r.item(), loss_d.item(), loss_front.item()]
            counted += 1

        totals /= max(counted, 1)
        history.append(
            dict(zip(("intensity", "range", "raydrop", "line_of_sight"), totals.tolist()))
        )
        log(
            f"  epoch {epoch + 1}/{epochs}: intensity {totals[0]:.5f}  range {totals[1]:.4f}  "
            f"raydrop {totals[2]:.4f}  ahead {totals[3]:.5f}  ({counted} views)"
        )

    with torch.no_grad():
        return (
            intensity.detach().cpu().numpy(),
            raydrop.detach().cpu().numpy(),
            torch.sigmoid(opacity_logit).detach().cpu().numpy(),
            history,
        )


#: Bumped when the cached array layout changes, so a stale file is refused.
RETURNS_CACHE_VERSION = 1


def save_returns(path, per_view: Sequence[ViewReturns]) -> None:
    """Cache the extracted returns of one object to a ``.npz``.

    Reading them back out of the dataset costs a camera image and a full lidar
    sweep per view -- ten minutes for a few hundred views -- while the fitting
    that follows takes a fraction of that. Refitting is the part worth iterating
    on, so the extraction is stored once.
    """
    from pathlib import Path

    if not per_view:
        raise ValueError("nothing to cache")
    fields = ("xyz", "intensity", "distance", "cos_incidence", "colour")
    payload = {
        name: np.concatenate([getattr(item.returns, name) for item in per_view])
        for name in fields
    }
    payload["hits_sensor"] = np.concatenate([item.hits_sensor for item in per_view])
    payload["viewmats"] = np.stack([item.viewmat for item in per_view])
    # One offset per view plus the end, so the concatenation splits back exactly.
    payload["offsets"] = np.cumsum([0] + [len(item.returns) for item in per_view])
    payload["version"] = np.asarray(RETURNS_CACHE_VERSION)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **payload)


def load_returns(path) -> list[ViewReturns]:
    """Read back what :func:`save_returns` wrote."""
    with np.load(str(path)) as cached:
        version = int(cached["version"])
        if version != RETURNS_CACHE_VERSION:
            raise ValueError(
                f"cache {path} is version {version}, this build writes "
                f"{RETURNS_CACHE_VERSION}; delete it to re-extract"
            )
        # Every `cached[name]` decompresses that whole array afresh. Reading them
        # once and slicing views into them keeps the file's worth of memory;
        # indexing inside the loop would decompress it once per view, and each
        # slice would pin its own full copy alive.
        offsets = cached["offsets"]
        arrays = {
            name: cached[name]
            for name in (
                "xyz",
                "intensity",
                "distance",
                "cos_incidence",
                "colour",
                "hits_sensor",
                "viewmats",
            )
        }

    views = []
    for index in range(len(offsets) - 1):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        views.append(
            ViewReturns(
                returns=ObjectReturns(
                    xyz=arrays["xyz"][lo:hi],
                    intensity=arrays["intensity"][lo:hi],
                    distance=arrays["distance"][lo:hi],
                    cos_incidence=arrays["cos_incidence"][lo:hi],
                    colour=arrays["colour"][lo:hi],
                    view_index=np.full(hi - lo, index, dtype=np.int64),
                ),
                hits_sensor=arrays["hits_sensor"][lo:hi],
                viewmat=arrays["viewmats"][index],
            )
        )
    return views


# --- geometry refinement ----------------------------------------------------


@dataclass
class GeometryFit:
    """A correction taking the reconstruction onto the lidar returns."""

    linear: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    yaw_deg: float
    scale: np.ndarray  # (3,) per-axis
    rms_before: float
    rms_after: float

    def to_dict(self) -> dict:
        return {
            "yaw_deg": self.yaw_deg,
            "scale_xyz": [float(v) for v in self.scale],
            "translation": [float(v) for v in self.translation],
            "rms_before_m": self.rms_before,
            "rms_after_m": self.rms_after,
        }


def _yaw_matrix(yaw_rad: float) -> np.ndarray:
    cos, sin = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def refine_to_lidar(
    means: np.ndarray,
    returns: ObjectReturns,
    *,
    max_points: int = 40000,
    trim: float = 0.8,
    reverse_trim: float = 0.5,
    reverse_weight: float = 1.0,
    max_scale_change: float = 1.4,
    seed: int = 0,
) -> GeometryFit:
    """Fit the reconstruction onto the object's own lidar returns.

    Aligning to the annotation box only pins the object as tightly as the box
    itself: a single-view SAM 3D reconstruction can sit inside a correct box and
    still be too wide, and ``scale_mode="iso"`` cannot correct one axis without
    the others. The lidar returns are the object's measured surface, pooled over
    every unoccluded view, so they constrain each axis and the heading directly.

    Solved for: yaw, a per-axis scale and a translation -- seven numbers, which
    a few thousand returns support comfortably, where a free 3x3 would start
    absorbing the reconstruction's own errors. Roll and pitch stay fixed: the
    annotation supplies them, and a vehicle on a road has little of either.

    Args:
        means: ``(N, 3)`` splat centres, in the box frame.
        returns: Pooled measurements for the same object, same frame.
        max_points: Subsample the returns to this many for the fit.
        trim: Fraction of the closest measurement->cloud correspondences to keep,
            so returns on surfaces the reconstruction never saw cannot drag the fit.
        reverse_trim: Same, for the cloud->measurement direction, which needs a
            harder trim because the lidar only sees the object's near side.
        reverse_weight: Weight of the cloud->measurement term.
        max_scale_change: Bound on each axis' scale factor. The box alignment is
            already close; anything beyond this is the fit running away, not a
            correction.
        seed: Subsampling seed.

    Returns:
        A :class:`GeometryFit`; apply it with
        :func:`~sam3d_objects.integrations.t4.asset.transform_ply`.
    """
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    points = returns.xyz
    if len(points) > max_points:
        points = points[rng.choice(len(points), max_points, replace=False)]
    splats = means
    if len(splats) > max_points:
        splats = splats[rng.choice(len(splats), max_points, replace=False)]

    tree_splats = cKDTree(splats)
    tree_points = cKDTree(points)
    centre = means.mean(axis=0)

    def trimmed(distance, fraction):
        kept = np.sort(distance)[: max(int(fraction * len(distance)), 1)]
        return float(np.sqrt(np.mean(kept**2)))

    def residual(params):
        yaw, log_scale, shift = params[0], params[1:4], params[4:7]
        rotation = _yaw_matrix(-yaw)
        # Measurements into the reconstruction's frame, so the splat tree is
        # built once; and the splats into the measurements' frame for the
        # reverse term.
        local = (points - shift - centre) @ rotation.T / np.exp(log_scale) + centre
        forward, _ = tree_splats.query(local, workers=-1)

        moved = (splats - centre) * np.exp(log_scale) @ _yaw_matrix(yaw).T + centre + shift
        backward, _ = tree_points.query(moved, workers=-1)

        # Both directions, or the fit has a runaway optimum: with only
        # measurement -> cloud, inflating the cloud lets every measurement find
        # a neighbour, and the scale grows without bound. The reverse term is
        # trimmed harder because the lidar only ever sees the near side, so a
        # large share of the splats legitimately has no measurement near it.
        return trimmed(forward, trim) + reverse_weight * trimmed(backward, reverse_trim)

    start = np.zeros(7)
    before = residual(start)
    span = np.log(max_scale_change)
    extent = float(np.ptp(means, axis=0).max())
    bounds = [(-np.pi / 12, np.pi / 12)] + [(-span, span)] * 3 + [(-0.2 * extent, 0.2 * extent)] * 3
    best = minimize(
        residual, start, method="Powell", bounds=bounds, options={"xtol": 1e-4, "ftol": 1e-4}
    )

    yaw, log_scale, shift = best.x[0], best.x[1:4], best.x[4:7]
    scale = np.exp(log_scale)
    linear = _yaw_matrix(yaw) @ np.diag(scale)
    # The scaling is about the cloud's own centre, so the offset keeps it there.
    translation = shift + centre - linear @ centre
    return GeometryFit(
        linear=linear,
        translation=translation,
        yaw_deg=float(np.degrees(yaw)),
        scale=scale,
        rms_before=before,
        rms_after=float(best.fun),
    )
