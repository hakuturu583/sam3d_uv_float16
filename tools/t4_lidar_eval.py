#!/usr/bin/env python3
"""Score a lidar-fitted asset against the returns the sensor actually measured.

Renders the asset at the pose of each held-out view and compares, cell by cell,
against the beams that came back there: intensity, first-return range, and
whether a beam returned at all. Two baselines sit alongside the fitted model so
the numbers mean something -- the object's mean reflectivity, and the
colour-only prediction the fit starts from. Anything that does not beat those
has learned nothing from the lidar.

    python tools/t4_lidar_eval.py \\
        --asset out/lidar_assets/000_bus_heldout.ply \\
        --cache out/lidar_masks/000_bus/returns.npz \\
        --report out/lidar_assets/000_bus_heldout.lidar.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam3d_objects.integrations.t4.asset import read_splats  # noqa: E402
from sam3d_objects.integrations.t4.lidar import (  # noqa: E402
    INTENSITY_SCALE,
    beam_grid,
    load_returns,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="ply carrying the lidar channels")
    parser.add_argument("--cache", required=True, help="returns npz the fit was built from")
    parser.add_argument(
        "--report", required=True, help="the fit's json, for the split and the model"
    )
    parser.add_argument(
        "--split",
        choices=("holdout", "train", "all"),
        default="holdout",
        help="which views to score",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="write the metrics as json")
    return parser


def load_asset(path):
    """The splats, with the lidar channels the scoring needs."""
    splats = read_splats(path)
    if not splats.has_lidar:
        raise SystemExit(f"{path} has no lidar channels; run tools/t4_lidar_attributes.py first")
    return dict(
        means=splats.means,
        quats=splats.quats,
        scales=splats.scales,
        colours=splats.colours,
        intensity=splats.intensity,
        raydrop=splats.raydrop_logit,
        opacity=splats.lidar_opacity,
    )


def render(asset, grid, feature, device):
    """Render one feature channel on a view's grid; returns it with depth and alpha."""
    import torch
    from splatad_kernel import lidar_rasterization

    raster, measured, target_range, target_intensity = grid.dense(device)
    tile_height = 8
    boundaries = torch.linspace(
        grid.min_elevation,
        grid.max_elevation,
        int(np.ceil(grid.height / tile_height)) + 1,
        device=device,
    )
    features = torch.stack(
        [
            torch.as_tensor(feature, dtype=torch.float32, device=device),
            torch.as_tensor(asset["raydrop"], dtype=torch.float32, device=device),
        ],
        dim=-1,
    )[None]
    rendered, alphas, _, meta = lidar_rasterization(
        means=torch.as_tensor(asset["means"], dtype=torch.float32, device=device),
        quats=torch.as_tensor(asset["quats"], dtype=torch.float32, device=device),
        scales=torch.as_tensor(asset["scales"], dtype=torch.float32, device=device),
        opacities=torch.as_tensor(asset["opacity"], dtype=torch.float32, device=device),
        lidar_features=features,
        velocities=None,
        viewmats=torch.as_tensor(grid.viewmat, dtype=torch.float32, device=device)[None],
        raster_pts=raster,
        tile_elevation_boundaries=boundaries,
        min_azimuth=-180.0,
        max_azimuth=180.0,
        min_elevation=grid.min_elevation,
        max_elevation=grid.max_elevation,
        n_elevation_channels=grid.height,
        azimuth_resolution=grid.azimuth_resolution,
        tile_width=8,
        tile_height=tile_height,
        compute_alpha_sum_until_points=False,
    )
    return (
        rendered[0, ..., 0],
        rendered[0, ..., 1],
        meta["median_depths"][0, ..., 0],
        alphas[0, ..., 0],
        measured,
        target_range,
        target_intensity,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    report = json.loads(Path(args.report).read_text())
    stride = int(report.get("holdout_stride") or 0)
    model = report["reflectance_model"]
    asset = load_asset(args.asset)
    per_view = load_returns(args.cache)

    grids = []
    for index, item in enumerate(per_view):
        returns = item.returns
        geometry = (
            np.maximum(returns.cos_incidence, 1e-3) ** model["cos_power"]
            / np.maximum(returns.distance, 1e-3) ** model["range_power"]
        )
        reflectance = np.clip(returns.intensity / np.maximum(geometry, 1e-9), 0, INTENSITY_SCALE)
        grid = beam_grid(returns, item.hits_sensor, item.viewmat, reflectance / INTENSITY_SCALE)
        if grid is None:
            continue
        held = stride > 1 and index % stride == 0
        if args.split == "all" or (args.split == "holdout") == held:
            grids.append(grid)
    if not grids:
        raise SystemExit("no views in this split; was the fit run with --holdout?")
    print(f"scoring {len(grids)} {args.split} view(s)")

    # Baselines: what the numbers would be with no lidar fitting at all.
    design = np.concatenate([asset["colours"], np.ones((len(asset["colours"]), 1))], 1)
    colour_only = (
        np.clip(design @ np.asarray(model["weights_rgb_bias"], np.float32), 0, INTENSITY_SCALE)
        / INTENSITY_SCALE
    )
    candidates = {
        "fitted": asset["intensity"],
        "colour_only": colour_only.astype(np.float32),
        "constant_mean": np.full_like(asset["intensity"], float(asset["intensity"].mean())),
    }

    errors = {name: [] for name in candidates}
    depth_error, alpha_hit, raydrop_ok, cells = [], [], [], 0
    for grid in grids:
        for name, feature in candidates.items():
            values, drop, depth, alpha, measured, target_r, target_i = render(
                asset, grid, feature, args.device
            )
            residual = (values - target_i)[measured]
            errors[name].append(residual.detach().cpu().numpy())
            if name == "fitted":
                seen = measured & (depth > 0)
                depth_error.append((depth - target_r)[seen].detach().cpu().numpy())
                alpha_hit.append((alpha[measured] > 0.3).float().mean().item())
                raydrop_ok.append(
                    (torch.sigmoid(drop[measured]) < 0.5).float().mean().item()
                )
                cells += int(measured.sum())

    def summarise(residuals):
        stacked = np.concatenate(residuals)
        return {
            "rmse_0_1": float(np.sqrt(np.mean(stacked**2))),
            "mae_0_1": float(np.mean(np.abs(stacked))),
            "rmse_sensor_units": float(np.sqrt(np.mean(stacked**2)) * INTENSITY_SCALE),
        }

    depth = np.concatenate(depth_error)
    metrics = {
        "split": args.split,
        "views": len(grids),
        "measured_cells": cells,
        "intensity": {name: summarise(values) for name, values in errors.items()},
        "range": {
            "median_abs_error_m": float(np.median(np.abs(depth))),
            "rmse_m": float(np.sqrt(np.mean(depth**2))),
            "within_0_2m": float(np.mean(np.abs(depth) < 0.2)),
        },
        "hit_rate": float(np.mean(alpha_hit)),
        "raydrop_recall": float(np.mean(raydrop_ok)),
    }

    print(f"\n{cells} measured cells over {len(grids)} views")
    print("intensity error (0-1 reflectance; sensor units in brackets)")
    for name, values in metrics["intensity"].items():
        print(
            f"  {name:14s} RMSE {values['rmse_0_1']:.4f}  MAE {values['mae_0_1']:.4f}"
            f"   [{values['rmse_sensor_units']:5.1f} / 255]"
        )
    print(
        f"range: median |error| {metrics['range']['median_abs_error_m']:.3f} m, "
        f"RMSE {metrics['range']['rmse_m']:.3f} m, "
        f"{100 * metrics['range']['within_0_2m']:.1f}% within 0.2 m"
    )
    print(
        f"a beam that returned in the data also returns here "
        f"{100 * metrics['hit_rate']:.1f}% of the time; "
        f"ray-drop agrees {100 * metrics['raydrop_recall']:.1f}%"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
