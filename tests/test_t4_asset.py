"""Splat ply I/O tests: the activations must survive a round trip.

Run on CPU; only `transform_ply` needs torch.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

plyfile = pytest.importorskip("plyfile")

from sam3d_objects.integrations.t4.asset import (
    LIDAR_CHANNELS,
    SH_DC_SCALE,
    read_splats,
    write_lidar_channels,
)


def write_ply(path, n=16, seed=0, lidar=False):
    """A ply in exactly the convention `Gaussian.save_ply` writes."""
    rng = np.random.default_rng(seed)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(3)] + ["opacity"]
    names += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    if lidar:
        names += list(LIDAR_CHANNELS)
    data = np.zeros(n, dtype=[(name, "f4") for name in names])
    for axis in "xyz":
        data[axis] = rng.uniform(-2, 2, n)
    for index in range(3):
        data[f"f_dc_{index}"] = rng.uniform(-1, 1, n)
        data[f"scale_{index}"] = np.log(rng.uniform(0.005, 0.05, n))
    data["opacity"] = rng.uniform(-4, 4, n)
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    for index in range(4):
        data[f"rot_{index}"] = quats[:, index]
    if lidar:
        for name in LIDAR_CHANNELS:
            data[name] = rng.uniform(0, 1, n)
    plyfile.PlyData([plyfile.PlyElement.describe(data, "vertex")]).write(str(path))
    return data


def test_read_undoes_the_stored_activations(tmp_path):
    path = tmp_path / "in.ply"
    raw = write_ply(path)
    splats = read_splats(path)

    assert len(splats) == len(raw)
    assert np.allclose(splats.means[:, 0], raw["x"])
    # scale is stored as a log, opacity as a logit, colour as an SH DC term.
    assert np.allclose(splats.scales[:, 0], np.exp(raw["scale_0"]), rtol=1e-5)
    assert np.allclose(splats.opacity, 1 / (1 + np.exp(-raw["opacity"])), rtol=1e-5)
    assert np.allclose(
        splats.colours[:, 0], np.clip(raw["f_dc_0"] * SH_DC_SCALE + 0.5, 0, 1), rtol=1e-5
    )
    assert np.allclose(np.linalg.norm(splats.quats, axis=1), 1.0, atol=1e-5)
    assert not splats.has_lidar


def test_read_picks_up_the_lidar_channels_when_present(tmp_path):
    path = tmp_path / "in.ply"
    raw = write_ply(path, lidar=True)
    splats = read_splats(path)

    assert splats.has_lidar
    assert np.allclose(splats.intensity, raw["intensity"])
    assert np.allclose(splats.raydrop_logit, raw["raydrop_logit"])
    assert np.allclose(splats.lidar_opacity, raw["lidar_opacity"])


def test_extent_ignores_the_transparent_haze(tmp_path):
    path = tmp_path / "in.ply"
    write_ply(path, n=64)
    splats = read_splats(path)
    # One splat far away, invisible: it must not stretch the extent.
    splats.means = np.vstack([splats.means, [[100.0, 0.0, 0.0]]]).astype(np.float32)
    splats.opacity = np.append(splats.opacity, 0.001).astype(np.float32)

    low, high = splats.extent()
    assert high[0] < 10.0


def test_write_adds_the_channels_and_keeps_everything_else(tmp_path):
    source = tmp_path / "in.ply"
    raw = write_ply(source, n=12)
    destination = tmp_path / "out.ply"

    values = np.linspace(0, 1, len(raw)).astype(np.float32)
    write_lidar_channels(source, destination, values, -values, 1 - values)

    written = plyfile.PlyData.read(str(destination)).elements[0]
    assert np.allclose(written["intensity"], values)
    assert np.allclose(written["raydrop_logit"], -values)
    assert np.allclose(written["lidar_opacity"], 1 - values)
    for name in ("x", "opacity", "scale_0", "rot_0", "f_dc_0"):
        assert np.allclose(written[name], raw[name]), f"{name} was disturbed"


def test_write_replaces_channels_rather_than_duplicating_them(tmp_path):
    """Refitting an asset must not grow a second `intensity` column."""
    source = tmp_path / "in.ply"
    raw = write_ply(source, n=8, lidar=True)
    destination = tmp_path / "out.ply"

    values = np.full(len(raw), 0.25, np.float32)
    write_lidar_channels(source, destination, values, values, values)

    written = plyfile.PlyData.read(str(destination)).elements[0]
    assert written.data.dtype.names.count("intensity") == 1
    assert np.allclose(written["intensity"], 0.25)


def test_write_rejects_a_length_mismatch(tmp_path):
    source = tmp_path / "in.ply"
    write_ply(source, n=8)
    with pytest.raises(ValueError, match="8 splats"):
        write_lidar_channels(source, tmp_path / "out.ply", np.zeros(3), np.zeros(3), np.zeros(3))
