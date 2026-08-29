#!/usr/bin/env python3
"""Reconstruct T4 dataset objects with SAM 3D and align them to their 3D boxes.

Reads a camera image and its annotated 3D bounding boxes with ``t4-devkit``
(v0.8.0), runs SAM 3D Objects on each masked object, then corrects the
reconstruction's scale and rotation against the annotation so that a car ends up
facing along the box's forward (+X) axis at its true metric size.

Example:
    python tools/t4_sam3d_align.py \\
        --data-root data/t4dataset/my_scene \\
        --config checkpoints/hf/pipeline.yaml \\
        --camera CAM_FRONT --category car \\
        --out-dir out/aligned
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
sys.path.insert(0, str(REPO_ROOT / "notebook"))

# Every option table is read from the library rather than retyped, so adding a
# mode cannot leave the CLI rejecting it. `pipeline` (and its torch import) stays
# deferred to main() so --help and --dry-run never load the model stack.
from sam3d_objects.integrations.t4.align import ROTATION_MODES, SCALE_MODES, Z_ALIGN_MODES
from sam3d_objects.integrations.t4.dataset import MASK_SOURCES
from sam3d_objects.integrations.t4.frames import FRAME_CHAIN, VIEWER_AXES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--data-root", required=True, help="T4 dataset root directory")
    dataset.add_argument("--revision", default=None, help="dataset revision (default: latest)")
    dataset.add_argument("--camera", default="CAM_FRONT", help="camera channel")
    dataset.add_argument("--sample-token", default=None, help="sample token (default: by index)")
    dataset.add_argument("--sample-index", type=int, default=0, help="index into t4.sample")
    dataset.add_argument(
        "--visibility",
        default=None,
        choices=["none", "partial", "most", "full"],
        help="drop boxes below this visibility level",
    )

    selection = parser.add_argument_group("object selection")
    selection.add_argument(
        "--category",
        action="append",
        default=None,
        help="category substring to keep, repeatable (default: every box)",
    )
    selection.add_argument(
        "--instance-token", default=None, help="reconstruct only this instance"
    )
    selection.add_argument("--max-distance", type=float, default=None, help="metres from camera")
    selection.add_argument("--min-area-px", type=float, default=0.0, help="min projected area")
    selection.add_argument("--min-lidar-points", type=int, default=None)
    selection.add_argument("--limit", type=int, default=None, help="max objects to reconstruct")

    mask = parser.add_argument_group("mask")
    mask.add_argument("--mask-source", default="auto", choices=MASK_SOURCES)
    mask.add_argument("--mask-dilate", type=int, default=0, help="dilate the mask, in pixels")
    mask.add_argument(
        "--mask-dir",
        default=None,
        help="directory of precomputed masks for --mask-source file "
        "(write them with tools/t4_sam3_masks.py)",
    )

    align = parser.add_argument_group("alignment")
    align.add_argument(
        "--rotation-mode",
        default="snap24",
        choices=ROTATION_MODES,
        help="snap24: square the object with the box (default). "
        "yaw: fix heading only, keep predicted roll/pitch. none: keep SAM 3D's rotation",
    )
    align.add_argument(
        "--scale-mode",
        default="iso",
        choices=SCALE_MODES,
        help="which box dimension refits the metric scale (default: iso)",
    )
    align.add_argument("--z-align", default="center", choices=Z_ALIGN_MODES)
    align.add_argument("--keep-translation", action="store_true")
    align.add_argument(
        "--extra-yaw-deg",
        type=float,
        default=0.0,
        help="manual heading offset applied after the correction; use 180 when "
        "SAM 3D reads a symmetric vehicle back-to-front",
    )
    align.add_argument("--percentile", type=float, default=1.0, help="outlier trim in %%")
    align.add_argument("--out-frame", default="box", choices=FRAME_CHAIN)
    align.add_argument(
        "--viewer-axes",
        default=None,
        choices=sorted(VIEWER_AXES),
        help="final axis swap for third party viewers; omit to keep the frame's own axes",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    model.add_argument("--seed", type=int, default=42)
    model.add_argument("--compile", action="store_true")

    densify = parser.add_argument_group("densification")
    densify.add_argument(
        "--densify-passes",
        type=int,
        default=0,
        help="fill the decoder's lattice holes with interpolated splats (0 = off)",
    )
    densify.add_argument(
        "--densify-coverage",
        type=float,
        default=2.0,
        help="fill a neighbour pair when it is longer than this many summed radii",
    )

    parser.add_argument("--out-dir", default="out/t4_aligned")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the selected boxes and write mask previews, without running SAM 3D",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from sam3d_objects.integrations.t4.dataset import load_camera_frame, load_t4, select_boxes

    t4 = load_t4(args.data_root, revision=args.revision)
    frame = load_camera_frame(
        t4,
        sample_token=args.sample_token,
        sample_index=args.sample_index,
        channel=args.camera,
        visibility=args.visibility,
    )
    boxes = select_boxes(
        frame,
        categories=args.category,
        min_area_px=args.min_area_px,
        max_distance=args.max_distance,
        min_lidar_points=args.min_lidar_points,
    )
    if args.instance_token is not None:
        boxes = [b for b in boxes if b.uuid == args.instance_token]
    if args.limit is not None:
        boxes = boxes[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("{}: {}x{}, {} object(s)".format(frame.channel, *frame.size, len(boxes)))

    if args.dry_run:
        return _dry_run(t4, frame, boxes, args, out_dir)

    from inference import Inference  # notebook/inference.py

    from sam3d_objects.integrations.t4.pipeline import reconstruct_box

    inference = Inference(args.config, compile=args.compile)

    for index, box in enumerate(boxes):
        label = box.semantic_label.name.replace("/", "_")
        stem = f"{index:03d}_{label}"
        print(f"[{index + 1}/{len(boxes)}] {label} at {np.round(box.position, 2)}", flush=True)

        result = reconstruct_box(
            inference,
            t4,
            frame,
            box,
            seed=args.seed,
            mask_source=args.mask_source,
            mask_dilate=args.mask_dilate,
            mask_dir=args.mask_dir,
            out_frame=args.out_frame,
            viewer_axes=args.viewer_axes,
            rotation_mode=args.rotation_mode,
            scale_mode=args.scale_mode,
            z_align=args.z_align,
            keep_translation=args.keep_translation,
            extra_yaw_deg=args.extra_yaw_deg,
            percentile=args.percentile,
        )
        if result is None:
            print("  skipped: no usable mask")
            continue

        if args.densify_passes > 0:
            from sam3d_objects.integrations.t4.densify import densify_gaussian

            before = len(result.gaussian.get_xyz)
            result.gaussian = densify_gaussian(
                result.gaussian,
                passes=args.densify_passes,
                coverage=args.densify_coverage,
            )
            print(f"  densified {before} -> {len(result.gaussian.get_xyz)} splats")

        result.save_ply(out_dir / f"{stem}.ply")
        result.save_report(out_dir / f"{stem}.json")
        report = result.alignment.report
        print(
            f"  yaw correction {report['yaw_correction_deg']:+.1f} deg, "
            f"snap {report['snap_angle_deg']:.1f} deg, "
            f"scale x{np.mean(report['scale_ratio_vs_sam3d']):.3f} vs SAM 3D "
            f"-> {out_dir / f'{stem}.ply'}"
        )
        # Drop the splats before the next object is reconstructed, so peak VRAM
        # never holds two full clouds plus the model.
        del result
    return 0


def _dry_run(t4, frame, boxes, args, out_dir: Path) -> int:  # noqa: D401
    """Report the selection and dump mask previews, without touching the model."""
    from PIL import Image

    from sam3d_objects.integrations.t4.dataset import box_mask

    summary = []
    for index, box in enumerate(boxes):
        mask = box_mask(t4, frame, box, source=args.mask_source, dilate=args.mask_dilate)
        pixels = int(mask.sum()) if mask is not None else 0
        label = box.semantic_label.name.replace("/", "_")
        summary.append(
            {
                "index": index,
                "category": box.semantic_label.name,
                "instance_token": box.uuid,
                "position_cam": np.round(np.asarray(box.position), 3).tolist(),
                "size_wlh": np.round(np.asarray(box.size), 3).tolist(),
                "distance_m": round(float(np.linalg.norm(box.position)), 2),
                "mask_pixels": pixels,
            }
        )
        if mask is not None and mask.any():
            preview = frame.image.copy()
            preview[~mask] = preview[~mask] // 4
            Image.fromarray(preview).save(out_dir / f"{index:03d}_{label}_mask.png")

    (out_dir / "selection.json").write_text(json.dumps(summary, indent=2))
    for row in summary:
        print(
            f"  [{row['index']:3d}] {row['category']:<24} "
            f"d={row['distance_m']:6.1f}m  wlh={row['size_wlh']}  mask={row['mask_pixels']}px"
        )
    print(f"wrote {out_dir / 'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
