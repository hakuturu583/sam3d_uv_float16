#!/usr/bin/env python3
"""Fill in a reconstructed object's lidar attributes from the dataset's returns.

SAM 3D gives an object its shape and colour; a lidar rasterizer also wants to
know how each splat reflects 905 nm light and how often a beam that hits it
comes back at all. This reads those out of the T4 dataset itself: every
unoccluded view of the object contributes the lidar points that land inside its
SAM 3 mask, a colour-to-reflectivity model is fitted to them, and the resulting
per-Gaussian intensity / ray-drop / opacity are optimised against the measured
beams through ``splatad_kernel``'s own lidar rasterizer -- with the geometry
frozen exactly as reconstructed.

    python tools/t4_lidar_attributes.py \\
        --data-root ~/.webauto/data/.../433a2328-... \\
        --asset out/webauto_assets/000_bus.ply \\
        --instance-token ea52bacb773b97142bc97b134b57a256 \\
        --mask-dir out/lidar_masks/000_bus \\
        --views-json views_000_bus.json \\
        --out out/lidar_assets/000_bus.ply

Masks come from ``tools/t4_sam3_masks.py --views-json``, which writes one per
view of the instance.
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

from sam3d_objects.integrations.t4.asset import (  # noqa: E402
    read_splats,
    transform_ply,
    write_lidar_channels,
)
from sam3d_objects.integrations.t4.lidar import (  # noqa: E402
    INTENSITY_SCALE,
    beam_grid,
    collect_returns,
    fit_reflectance,
    load_returns,
    refine_to_lidar,
    save_returns,
    train_lidar_features,
    unoccluded_views,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = parser.add_argument_group("data")
    data.add_argument("--data-root", required=True, help="T4 dataset root directory")
    data.add_argument("--revision", default=None, help="dataset revision (default: latest)")
    data.add_argument("--asset", required=True, help="aligned splat ply, in the box frame")
    data.add_argument("--instance-token", required=True, help="the object to read returns from")
    data.add_argument("--mask-dir", required=True, help="SAM 3 masks for this instance")
    data.add_argument(
        "--views-json",
        default=None,
        help="views to train on (default: rescan the dataset for unoccluded ones)",
    )
    data.add_argument("--max-views", type=int, default=None, help="cap the number of views")
    data.add_argument(
        "--cache",
        default=None,
        help="npz of extracted returns (default: <mask-dir>/returns.npz). Reading the "
        "dataset costs an image and a lidar sweep per view; refitting does not.",
    )
    data.add_argument(
        "--refresh-cache", action="store_true", help="re-extract even if the cache exists"
    )

    fit = parser.add_argument_group("fitting")
    fit.add_argument("--epochs", type=int, default=6)
    fit.add_argument("--learning-rate", type=float, default=0.02)
    fit.add_argument(
        "--opacity-init",
        type=float,
        default=0.99,
        help="starting lidar opacity; a beam reaching a surface almost always returns",
    )
    fit.add_argument(
        "--refine-geometry",
        action="store_true",
        help="refit yaw and per-axis scale onto the lidar returns before training; "
        "the box alignment only constrains the object as tightly as the box does",
    )
    fit.add_argument(
        "--holdout",
        type=float,
        default=0.0,
        help="fraction of views kept out of training, for tools/t4_lidar_eval.py. "
        "Every Nth view is held out, so the split is reproducible from this number alone.",
    )
    fit.add_argument("--device", default="cuda")

    parser.add_argument("--out", required=True, help="ply to write, with the lidar channels added")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from sam3d_objects.integrations.t4.dataset import load_t4

    splats = read_splats(args.asset)
    print(f"{args.asset}: {len(splats)} splats")

    t4 = load_t4(args.data_root, revision=args.revision)
    if args.views_json:
        raw = json.loads(Path(args.views_json).read_text())
        views = [
            type("V", (), dict(sample_token=v["sample_token"], camera=v["camera"]))()
            for v in raw
        ]
    else:
        views = unoccluded_views(t4, args.instance_token)
    if args.max_views:
        views = views[: args.max_views]
    print(f"{len(views)} unoccluded view(s)")

    cache = Path(args.cache) if args.cache else Path(args.mask_dir) / "returns.npz"
    if cache.exists() and not args.refresh_cache:
        per_view = load_returns(cache)
        print(f"loaded {len(per_view)} view(s) of returns from {cache}")
    else:
        per_view = []
        for index, view in enumerate(views):
            collected = collect_returns(
                t4, view, args.instance_token, args.mask_dir, view_index=index
            )
            if collected is not None and len(collected.returns) > 0:
                per_view.append(collected)
            if (index + 1) % 50 == 0:
                print(
                    f"  [{index + 1}/{len(views)}] {len(per_view)} view(s) with returns", flush=True
                )
        if not per_view:
            print("no lidar returns landed inside the masks; nothing to fit")
            return 1
        save_returns(cache, per_view)
        print(f"cached {len(per_view)} view(s) of returns to {cache}")

    pooled = per_view[0].returns
    for item in per_view[1:]:
        pooled = pooled.concat(item.returns)
    print(
        f"{len(pooled)} returns from {len(per_view)} views; "
        f"intensity {pooled.intensity.min():.0f}-{pooled.intensity.max():.0f} "
        f"(mean {pooled.intensity.mean():.1f}), range {pooled.distance.min():.1f}-"
        f"{pooled.distance.max():.1f} m"
    )

    geometry_fit = None
    if args.refine_geometry:
        geometry_fit = refine_to_lidar(splats.means, pooled)
        print(
            f"geometry refit on the returns: yaw {geometry_fit.yaw_deg:+.2f} deg, "
            f"scale {np.round(geometry_fit.scale, 3).tolist()}, "
            f"nearest-return RMS {geometry_fit.rms_before:.3f} -> {geometry_fit.rms_after:.3f} m"
        )
        refined = Path(args.out).with_suffix(".geom.ply")
        refined.parent.mkdir(parents=True, exist_ok=True)
        transform_ply(args.asset, refined, geometry_fit.linear, geometry_fit.translation)
        asset_for_training = str(refined)
        splats = read_splats(asset_for_training)
    else:
        asset_for_training = args.asset

    model = fit_reflectance(pooled)
    print(
        f"reflectance model: cos^{model.cos_power:g} / R^{model.range_power:g}, "
        f"R2={model.r2:.3f} on {model.n_points} points, "
        f"weights RGB+bias={np.round(model.weights, 2).tolist()}"
    )

    # Geometry divided out of every measurement: what is left is the intrinsic,
    # view-independent reflectivity the rendered feature has to reproduce.
    grids = []
    for item in per_view:
        returns = item.returns
        geometry = (
            np.maximum(returns.cos_incidence, 1e-3) ** model.cos_power
            / np.maximum(returns.distance, 1e-3) ** model.range_power
        )
        reflectance = np.clip(returns.intensity / np.maximum(geometry, 1e-9), 0, INTENSITY_SCALE)
        grid = beam_grid(
            returns, item.hits_sensor, item.viewmat, reflectance / INTENSITY_SCALE
        )
        if grid is not None:
            grids.append(grid)
    stride = int(round(1.0 / args.holdout)) if args.holdout > 0 else 0
    if stride > 1:
        held = [g for i, g in enumerate(grids) if i % stride == 0]
        grids = [g for i, g in enumerate(grids) if i % stride != 0]
        print(f"holding out every {stride}th view: {len(held)} for evaluation")
    print(f"{len(grids)} beam grids, {sum(g.n_measured for g in grids)} supervised cells")

    initial = model.predict(splats.colours) / INTENSITY_SCALE
    print(f"colour-predicted reflectivity: mean {initial.mean():.3f}, sd {initial.std():.3f}")

    intensity, raydrop, opacity, history = train_lidar_features(
        (splats.means, splats.quats, splats.scales),
        grids,
        initial,
        camera_opacity=splats.opacity,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        opacity_init=args.opacity_init,
        device=args.device,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_lidar_channels(asset_for_training, out, intensity, raydrop, opacity)
    report = {
        "asset": str(args.asset),
        "instance_token": args.instance_token,
        "views_used": len(grids),
        "holdout_stride": stride,
        "returns": int(len(pooled)),
        "geometry_fit": None if geometry_fit is None else geometry_fit.to_dict(),
        "reflectance_model": model.to_dict(),
        "intensity_scale": INTENSITY_SCALE,
        "training": history,
        "final": {
            "intensity_mean": float(intensity.mean()),
            "intensity_sd": float(intensity.std()),
            "raydrop_prob_mean": float(1 / (1 + np.exp(-raydrop)).mean()),
            "lidar_opacity_mean": float(opacity.mean()),
        },
    }
    out.with_suffix(".lidar.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out} and {out.with_suffix('.lidar.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
