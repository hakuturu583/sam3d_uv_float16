#!/usr/bin/env python3
"""Segment T4 dataset objects with SAM 3, for SAM 3D to reconstruct from.

Without 2D annotations a T4 object's only mask is the convex hull of its
projected 3D box, which hands SAM 3D a slab of road, sky and whatever stands
behind the vehicle. SAM 3 turns that box into an actual instance mask: the box
becomes the prompt, and the returned mask is written as
``<sample_data_token>__<instance_token>.png`` -- the name
:func:`~sam3d_objects.integrations.t4.dataset.mask_key` builds, so
``t4_sam3d_align.py --mask-source file --mask-dir ...`` picks it straight up.

SAM 3 pins ``timm>=1.0.17`` while SAM 3D pins ``timm==0.9.16``, so the two do
not share an environment. Run this in a SAM 3 one::

    uv venv --python 3.10 .venv-sam3
    VIRTUAL_ENV=$PWD/.venv-sam3 uv pip install \\
        --extra-index-url https://download.pytorch.org/whl/cu121 \\
        torch==2.5.1+cu121 torchvision==0.20.1+cu121 sam3 \\
        opencv-python psutil setuptools "t4-devkit @ git+https://github.com/tier4/t4-devkit@v0.8.0"

    .venv-sam3/bin/python tools/t4_sam3_masks.py \\
        --data-root data/t4dataset/my_scene --camera CAM_FRONT \\
        --sample-index 300 --category bus --out-dir out/sam3_masks

Only ``numpy``/``cv2``/``t4-devkit`` are imported from this repository, never the
model stack, so the SAM 3 environment stays free of SAM 3D's dependencies.
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

from sam3d_objects.integrations.t4.dataset import mask_key  # noqa: E402
from sam3d_objects.integrations.t4.views import fully_inside, projected_corners  # noqa: E402


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
        "--views-json",
        default=None,
        help="JSON list of {sample_index, camera} to segment in one model load, "
        "instead of the single frame named by --sample-index/--camera",
    )
    dataset.add_argument(
        "--visibility",
        default=None,
        choices=["none", "partial", "most", "full"],
        help="drop boxes below this visibility level",
    )

    selection = parser.add_argument_group("object selection")
    selection.add_argument(
        "--category", action="append", default=None, help="category substring, repeatable"
    )
    selection.add_argument("--max-distance", type=float, default=None, help="metres from camera")
    selection.add_argument("--min-area-px", type=float, default=0.0, help="min projected area")
    selection.add_argument("--limit", type=int, default=None, help="max objects to segment")
    selection.add_argument(
        "--instance-token", default=None, help="segment only this instance"
    )
    selection.add_argument(
        "--fully-visible",
        action="store_true",
        help="keep only boxes whose eight corners all project inside the frame",
    )
    selection.add_argument(
        "--margin", type=int, default=25, help="border --fully-visible keeps clear, in pixels"
    )

    segmentation = parser.add_argument_group("segmentation")
    segmentation.add_argument(
        "--containment",
        type=float,
        default=0.6,
        help="reject a mask with less than this fraction of itself inside the box hull",
    )
    segmentation.add_argument(
        "--min-area-frac",
        type=float,
        default=0.02,
        help="reject a mask smaller than this fraction of the prompt box",
    )
    segmentation.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="reject a mask SAM 3 predicts a lower IoU than this for",
    )
    segmentation.add_argument("--device", default="cuda")

    parser.add_argument("--out-dir", default="out/sam3_masks")
    parser.add_argument(
        "--preview", action="store_true", help="also write a cut-out RGBA preview per mask"
    )
    return parser


def choose_mask(masks, scores, hull, prompt_area, *, containment, min_area_frac, min_score):
    """Pick the SAM 3 candidate that best explains the annotated box.

    SAM 3 returns several nested readings of one prompt -- a wheel, the cabin,
    the whole vehicle. The box says which scale is wanted: keep the candidates
    that stay inside its hull and are not a speck, then take the largest, since
    the whole object is what SAM 3D has to reconstruct.

    ``min_score`` gates that on SAM 3's own predicted IoU first. Size alone picks
    the ragged half-road blob a low-confidence prompt returns, and feeding that
    to SAM 3D is worse than not reconstructing the object at all.
    """
    best, best_key = None, None
    for mask, score in zip(masks, scores):
        if float(score) < min_score:
            continue
        mask = mask.astype(bool)
        area = int(mask.sum())
        if area < min_area_frac * prompt_area:
            continue
        inside = float((mask & hull).sum()) / area
        if inside < containment:
            continue
        key = (area, float(score))
        if best_key is None or key > best_key:
            best, best_key = (mask, float(score), inside), key
    return best


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3d_objects.integrations.t4.dataset import load_camera_frame, load_t4

    t4 = load_t4(args.data_root, revision=args.revision)

    if args.views_json:
        views = json.loads(Path(args.views_json).read_text())
    else:
        views = [{"sample_index": args.sample_index, "camera": args.camera}]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(device=args.device, enable_inst_interactivity=True)
    processor = Sam3Processor(model, device=args.device)

    report = []
    for position, view in enumerate(views):
        frame = load_camera_frame(
            t4,
            sample_token=view.get("sample_token", args.sample_token if not args.views_json else None),
            sample_index=view["sample_index"],
            channel=view["camera"],
            visibility=args.visibility,
        )
        report += segment_frame(t4, frame, args, model, processor, out_dir)
        if args.views_json and position % 25 == 0:
            print(f"  [{position + 1}/{len(views)}] {len(report)} mask(s) so far", flush=True)

    (out_dir / "masks.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {len(report)} mask(s) to {out_dir}")
    return 0


def segment_frame(t4, frame, args, model, processor, out_dir) -> list:
    """Segment every selected box in one image, writing a mask png for each."""
    import cv2
    from PIL import Image

    from sam3d_objects.integrations.t4.dataset import mask_key, select_boxes

    boxes = select_boxes(
        frame,
        categories=args.category,
        min_area_px=args.min_area_px,
        max_distance=args.max_distance,
    )
    if args.instance_token is not None:
        boxes = [b for b in boxes if b.uuid == args.instance_token]

    width, height = frame.size
    if args.fully_visible:
        kept = []
        for box in boxes:
            uv = projected_corners(frame, box)
            if uv is None:
                continue
            if not fully_inside(uv, (width, height), args.margin):
                continue
            kept.append(box)
        boxes = kept
    if args.limit is not None:
        boxes = boxes[: args.limit]

    print(f"{frame.channel}: {width}x{height}, {len(boxes)} object(s)")
    if not boxes:
        return []

    # The image is encoded once; every box below is decoded against that state.
    # `set_image` reads the size off `shape[-2:]`, so hand it a PIL image rather
    # than an (H, W, 3) array, whose last two axes are width and channels.
    state = processor.set_image(Image.fromarray(frame.image))

    report = []
    for box in boxes:
        label = box.semantic_label.name
        uv = projected_corners(frame, box)
        if uv is None:
            print(f"  {label}: behind the camera, skipped")
            continue

        hull = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2)).astype(np.int32), 1)
        hull = hull.astype(bool)

        prompt = np.array(
            [
                max(uv[:, 0].min(), 0),
                max(uv[:, 1].min(), 0),
                min(uv[:, 0].max(), width - 1),
                min(uv[:, 1].max(), height - 1),
            ],
            dtype=np.float32,
        )
        prompt_area = float((prompt[2] - prompt[0]) * (prompt[3] - prompt[1]))
        masks, scores, _ = model.predict_inst(state, box=prompt, multimask_output=True)

        chosen = choose_mask(
            np.asarray(masks).reshape(-1, height, width),
            np.asarray(scores).reshape(-1),
            hull,
            prompt_area,
            containment=args.containment,
            min_area_frac=args.min_area_frac,
            min_score=args.min_score,
        )
        if chosen is None:
            best = float(np.max(np.asarray(scores))) if len(scores) else float("nan")
            print(
                f"  {label}: no usable candidate (best predicted IoU {best:.3f}), skipped"
            )
            continue
        mask, score, inside = chosen

        stem = mask_key(frame, box)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / f"{stem}.png")
        area, hull_area = int(mask.sum()), int(hull.sum())
        print(
            f"  {label}: {area}px (hull {hull_area}px, {100 * area / hull_area:.0f}% of it), "
            f"score {score:.3f}, {100 * inside:.0f}% inside -> {stem}.png"
        )
        report.append(
            dict(
                stem=stem,
                category=label,
                sample_token=frame.sample_token,
                sample_data_token=frame.sample_data_token,
                camera=frame.channel,
                instance_token=box.uuid,
                prompt_box=[float(v) for v in prompt],
                mask_px=area,
                hull_px=hull_area,
                score=score,
                inside_hull=inside,
            )
        )

        if args.preview:
            ys, xs = np.nonzero(mask)
            rgba = np.dstack([frame.image, (mask * 255).astype(np.uint8)])
            Image.fromarray(rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]).save(
                out_dir / f"{stem}_preview.png"
            )

    return report


if __name__ == "__main__":
    raise SystemExit(main())
