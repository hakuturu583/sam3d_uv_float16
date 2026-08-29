#!/usr/bin/env python3
"""Take a T4 dataset from raw frames to lidar-ready Gaussian splat assets.

Six stages, each writing into ``--work-dir`` and each skippable, so a change to
one does not force the ones before it to run again:

===========  ====================================================================
``scan``     Find every object seen unoccluded and whole, rank the instances,
             and write ``targets.json`` -- the reconstruction frame per object
             plus every frame the lidar fit may train on.
``masks``    Segment those frames with SAM 3. Runs in its own interpreter:
             SAM 3 pins ``timm>=1.0.17`` and SAM 3D pins ``timm==0.9.16``.
``build``    Reconstruct each target with SAM 3D from its best frame and align
             it to the annotated 3D box.
``lidar``    Refit the geometry onto the object's own lidar returns, then fit
             per-Gaussian intensity / ray-drop / opacity to them.
``eval``     Score the fit against held-out views.
``video``    Turn each asset in front of a fixed HDL-64E and record the sweep.
===========  ====================================================================

    python tools/t4_pipeline.py --data-root ~/.webauto/data/.../433a2328-... \\
        --work-dir out/pipeline --categories car truck bus --limit 4

Stages that need the GPU run one object at a time; nothing here is parallel,
because a single reconstruction already fills a 24 GB card.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam3d_objects.integrations.t4.views import (  # noqa: E402
    CAMERAS,
    ObjectView,
    best_per_instance,
    scan_views,
)

STAGES = ("scan", "masks", "build", "lidar", "eval", "video")


@dataclass
class Target:
    """One object the pipeline will turn into an asset."""

    name: str
    instance_token: str
    category: str
    best_view: ObjectView
    train_views: list[ObjectView]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instance_token": self.instance_token,
            "category": self.category,
            "best_view": self.best_view.to_dict(),
            "train_views": [view.to_dict() for view in self.train_views],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Target":
        return cls(
            name=payload["name"],
            instance_token=payload["instance_token"],
            category=payload["category"],
            best_view=ObjectView.from_dict(payload["best_view"]),
            train_views=[ObjectView.from_dict(v) for v in payload["train_views"]],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True, help="T4 dataset root directory")
    parser.add_argument("--revision", default=None, help="dataset revision (default: latest)")
    parser.add_argument("--work-dir", default="out/pipeline", help="where every stage writes")
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"comma-separated subset of {','.join(STAGES)}",
    )

    selection = parser.add_argument_group("selection")
    selection.add_argument("--categories", nargs="*", default=["car", "truck", "bus"])
    selection.add_argument("--cameras", nargs="*", default=list(CAMERAS))
    selection.add_argument("--limit", type=int, default=4, help="how many objects to build")
    selection.add_argument("--margin", type=int, default=25, help="frame border, pixels")
    selection.add_argument(
        "--min-area-px",
        type=float,
        default=100_000.0,
        help="an object must project at least this large in its best frame",
    )
    selection.add_argument("--min-lidar-pts", type=int, default=1)
    selection.add_argument(
        "--max-train-views", type=int, default=None, help="cap the views the lidar fit uses"
    )

    run = parser.add_argument_group("execution")
    run.add_argument(
        "--sam3-python",
        default=str(REPO_ROOT / ".venv-sam3/bin/python"),
        help="interpreter with SAM 3 installed, for the masks stage",
    )
    run.add_argument("--python", default=sys.executable, help="interpreter for every other stage")
    run.add_argument("--config", default="checkpoints/hf/pipeline.yaml", help="SAM 3D config")
    run.add_argument("--epochs", type=int, default=6, help="lidar fitting epochs")
    run.add_argument("--holdout", type=float, default=0.2, help="views kept back for eval")
    run.add_argument("--frames", type=int, default=120, help="turntable frames")
    run.add_argument("--device", default="cuda")
    return parser


def run_step(command: list[str], label: str) -> None:
    """Run one stage's subprocess, failing loudly rather than half-writing."""
    print(f"\n$ {' '.join(str(c) for c in command)}", flush=True)
    result = subprocess.run([str(c) for c in command], cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


# --- stages -----------------------------------------------------------------


def stage_scan(args, work: Path) -> list[Target]:
    """Rank the objects worth reconstructing and record the frames each needs."""
    from sam3d_objects.integrations.t4.dataset import load_t4

    t4 = load_t4(args.data_root, revision=args.revision)

    def progress(index, total, found):
        if index % 100 == 0:
            print(f"  sample {index}/{total}: {found} clean view(s)", flush=True)

    views = scan_views(
        t4,
        cameras=args.cameras,
        categories=args.categories,
        margin=args.margin,
        min_lidar_pts=args.min_lidar_pts,
        progress=progress,
    )
    print(f"{len(views)} clean views of {len({v.instance_token for v in views})} instances")

    targets = []
    for rank, best in enumerate(best_per_instance(views)):
        if best.area_px < args.min_area_px:
            continue
        train = [v for v in views if v.instance_token == best.instance_token]
        if args.max_train_views:
            train = train[: args.max_train_views]
        targets.append(
            Target(
                name=f"{len(targets):03d}_{best.category.replace('/', '_')}",
                instance_token=best.instance_token,
                category=best.category,
                best_view=best,
                train_views=train,
            )
        )
        if len(targets) >= args.limit:
            break

    payload = [target.to_dict() for target in targets]
    (work / "targets.json").write_text(json.dumps(payload, indent=2))
    for target in targets:
        print(
            f"  {target.name}: {target.category} at {target.best_view.distance_m:.1f} m in "
            f"{target.best_view.camera} (sample {target.best_view.sample_index}), "
            f"{len(target.train_views)} training view(s)"
        )
    print(f"wrote {work / 'targets.json'}")
    return targets


def stage_masks(args, work: Path, targets: list[Target]) -> None:
    """Segment every frame the later stages read, with SAM 3."""
    if not Path(args.sam3_python).exists():
        raise SystemExit(
            f"{args.sam3_python} does not exist; see tools/t4_sam3_masks.py for the "
            "environment SAM 3 needs"
        )
    for target in targets:
        views = work / "views" / f"{target.name}.json"
        views.parent.mkdir(parents=True, exist_ok=True)
        views.write_text(
            json.dumps(
                [
                    {"sample_index": v.sample_index, "sample_token": v.sample_token, "camera": v.camera}
                    for v in target.train_views
                ],
                indent=1,
            )
        )
        run_step(
            [
                args.sam3_python,
                "tools/t4_sam3_masks.py",
                "--data-root", args.data_root,
                "--views-json", views,
                "--instance-token", target.instance_token,
                "--out-dir", work / "masks" / target.name,
            ],
            f"masks for {target.name}",
        )


def stage_build(args, work: Path, targets: list[Target]) -> None:
    """Reconstruct each object from its best frame, aligned to its box."""
    for target in targets:
        view = target.best_view
        run_step(
            [
                args.python,
                "tools/t4_sam3d_align.py",
                "--data-root", args.data_root,
                "--camera", view.camera,
                "--sample-token", view.sample_token,
                "--config", args.config,
                "--mask-source", "file",
                "--mask-dir", work / "masks" / target.name,
                "--instance-token", target.instance_token,
                "--out-dir", work / "assets" / target.name,
            ],
            f"reconstruction of {target.name}",
        )


def stage_lidar(args, work: Path, targets: list[Target]) -> None:
    """Give each asset the channels a lidar rasterizer reads."""
    for target in targets:
        asset = _asset_path(work, target)
        run_step(
            [
                args.python,
                "tools/t4_lidar_attributes.py",
                "--data-root", args.data_root,
                "--asset", asset,
                "--instance-token", target.instance_token,
                "--mask-dir", work / "masks" / target.name,
                "--views-json", work / "views" / f"{target.name}.json",
                "--refine-geometry",
                "--holdout", args.holdout,
                "--epochs", args.epochs,
                "--device", args.device,
                "--out", work / "lidar" / f"{target.name}.ply",
            ],
            f"lidar fit for {target.name}",
        )


def stage_eval(args, work: Path, targets: list[Target]) -> None:
    """Score each fit on the views it never saw."""
    for target in targets:
        run_step(
            [
                args.python,
                "tools/t4_lidar_eval.py",
                "--asset", work / "lidar" / f"{target.name}.ply",
                "--cache", work / "masks" / target.name / "returns.npz",
                "--report", work / "lidar" / f"{target.name}.lidar.json",
                "--split", "holdout",
                "--device", args.device,
                "--out", work / "eval" / f"{target.name}.json",
            ],
            f"evaluation of {target.name}",
        )


def stage_video(args, work: Path, targets: list[Target]) -> None:
    """Record each asset turning in front of a fixed HDL-64E."""
    for target in targets:
        run_step(
            [
                args.python,
                "tools/t4_lidar_turntable.py",
                "--asset", work / "lidar" / f"{target.name}.ply",
                "--out", work / "video" / f"{target.name}.mp4",
                "--frames", args.frames,
                "--device", args.device,
            ],
            f"turntable for {target.name}",
        )


def _asset_path(work: Path, target: Target) -> Path:
    """The ply the build stage produced, whatever index the aligner gave it."""
    directory = work / "assets" / target.name
    candidates = sorted(p for p in directory.glob("*.ply") if not p.name.endswith(".geom.ply"))
    if not candidates:
        raise SystemExit(f"no reconstruction in {directory}; run the build stage first")
    return candidates[0]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stage(s) {sorted(unknown)}; pick from {', '.join(STAGES)}")

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    targets_file = work / "targets.json"
    if "scan" in stages:
        targets = stage_scan(args, work)
    else:
        if not targets_file.exists():
            raise SystemExit(f"{targets_file} does not exist; run the scan stage first")
        targets = [Target.from_dict(t) for t in json.loads(targets_file.read_text())]
        print(f"{len(targets)} target(s) from {targets_file}")

    for stage, function in (
        ("masks", stage_masks),
        ("build", stage_build),
        ("lidar", stage_lidar),
        ("eval", stage_eval),
        ("video", stage_video),
    ):
        if stage in stages:
            print(f"\n=== {stage} ===")
            function(args, work, targets)

    print(f"\ndone; everything is under {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
