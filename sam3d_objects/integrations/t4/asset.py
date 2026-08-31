"""Read and write the splat plys the pipeline passes between its stages.

A SAM 3D ply stores *activated* values -- the log of the filtered radius, the
biased quaternion, the SH DC term rather than a colour -- and every tool that
touched one had grown its own copy of the conversions. They are here once, so a
change to the convention lands in one place.

The lidar channels this pipeline adds (``intensity``, ``raydrop_logit``,
``lidar_opacity``) ride alongside the visual ones rather than replacing them: a
surface can be opaque to 905 nm and translucent to the eye, glass most of all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "LIDAR_CHANNELS",
    "SH_DC_SCALE",
    "Splats",
    "read_splats",
    "transform_ply",
    "write_lidar_channels",
]

#: Zeroth-order spherical harmonic, the DC term's scale.
SH_DC_SCALE = 0.28209479177387814

#: The per-splat fields a lidar rasterizer needs, on top of the visual ones.
LIDAR_CHANNELS = ("intensity", "raydrop_logit", "lidar_opacity")


@dataclass
class Splats:
    """One splat cloud, with activations undone.

    Attributes:
        means: ``(N, 3)`` centres.
        quats: ``(N, 4)`` normalised rotations, ``(w, x, y, z)``.
        scales: ``(N, 3)`` radii in metres.
        colours: ``(N, 3)`` RGB in 0-1.
        opacity: ``(N,)`` visual opacity in 0-1.
        intensity: ``(N,)`` lidar reflectivity in 0-1, or ``None``.
        raydrop_logit: ``(N,)`` ray-drop logit, or ``None``.
        lidar_opacity: ``(N,)`` opacity to a lidar beam, or ``None``.
    """

    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    colours: np.ndarray
    opacity: np.ndarray
    intensity: np.ndarray | None = None
    raydrop_logit: np.ndarray | None = None
    lidar_opacity: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.means)

    @property
    def has_lidar(self) -> bool:
        return self.intensity is not None

    def extent(self, opacity_floor: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """``(low, high)`` corners of the cloud, ignoring near-transparent splats."""
        solid = self.means[self.opacity > opacity_floor]
        if len(solid) == 0:
            solid = self.means
        return solid.min(axis=0), solid.max(axis=0)


def read_splats(path) -> Splats:
    """Load a SAM 3D ply, undoing the stored activations."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path)).elements[0]
    names = set(vertex.data.dtype.names)

    def column(name):
        return np.asarray(vertex[name], np.float32)

    quats = np.stack([column(f"rot_{i}") for i in range(4)], 1)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    lidar = {name: column(name) for name in LIDAR_CHANNELS} if set(LIDAR_CHANNELS) <= names else {}
    return Splats(
        means=np.stack([column("x"), column("y"), column("z")], 1),
        quats=quats,
        # `save_ply` writes log of the *filtered* radius, so exp is the radius.
        scales=np.exp(np.stack([column(f"scale_{i}") for i in range(3)], 1)),
        colours=np.clip(
            np.stack([column(f"f_dc_{i}") for i in range(3)], 1) * SH_DC_SCALE + 0.5, 0.0, 1.0
        ),
        opacity=1.0 / (1.0 + np.exp(-column("opacity"))),
        intensity=lidar.get("intensity"),
        raydrop_logit=lidar.get("raydrop_logit"),
        lidar_opacity=lidar.get("lidar_opacity"),
    )


def write_lidar_channels(source, destination, intensity, raydrop_logit, lidar_opacity) -> None:
    """Copy a splat ply, adding (or replacing) the channels a lidar rasterizer reads."""
    from plyfile import PlyData, PlyElement

    vertex = PlyData.read(str(source)).elements[0]
    if len(vertex) != len(intensity):
        raise ValueError(f"ply has {len(vertex)} splats, got {len(intensity)} values")

    extra = dict(
        zip(
            LIDAR_CHANNELS,
            (np.asarray(a, np.float32) for a in (intensity, raydrop_logit, lidar_opacity)),
        )
    )
    existing = [name for name in vertex.data.dtype.names if name not in extra]
    dtype = [(name, vertex.data.dtype[name].str) for name in existing]
    dtype += [(name, "f4") for name in extra]

    merged = np.empty(len(vertex), dtype=dtype)
    for name in existing:
        merged[name] = vertex.data[name]
    for name, values in extra.items():
        merged[name] = values

    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")]).write(str(destination))


def transform_ply(source, destination, linear, translation) -> None:
    """Rewrite a splat ply through an affine map, covariances carried with it.

    The round trip goes back through the stored activations rather than the raw
    parameters, so the file stays in exactly the convention SAM 3D wrote.
    """
    import torch
    from plyfile import PlyData, PlyElement

    from .gaussian_ops import transform_splats

    vertex = PlyData.read(str(source)).elements[0]
    xyz = torch.tensor(np.stack([vertex["x"], vertex["y"], vertex["z"]], 1).astype(np.float32))
    quats = torch.tensor(np.stack([vertex[f"rot_{i}"] for i in range(4)], 1).astype(np.float32))
    quats = torch.nn.functional.normalize(quats, dim=-1)
    scales = torch.tensor(
        np.exp(np.stack([vertex[f"scale_{i}"] for i in range(3)], 1).astype(np.float32))
    )

    new_xyz, new_quats, new_scales = transform_splats(
        xyz, quats, scales, np.asarray(linear, np.float64), np.asarray(translation, np.float64)
    )

    merged = vertex.data.copy()
    for index, axis in enumerate("xyz"):
        merged[axis] = new_xyz[:, index].numpy()
    for index in range(4):
        merged[f"rot_{index}"] = new_quats[:, index].numpy()
    for index in range(3):
        merged[f"scale_{index}"] = np.log(np.maximum(new_scales[:, index].numpy(), 1e-12))

    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")]).write(str(destination))
