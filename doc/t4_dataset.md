# T4 dataset (tier4/t4-devkit v0.8.0) integration

Reconstruct annotated objects from a [T4 dataset](https://github.com/tier4/t4-devkit)
camera image with SAM 3D Objects, then correct the reconstruction's **scale** and
**rotation** against the 3D bounding box so a car ends up facing forward at its
true metric size.

```bash
pip install "t4-devkit @ git+https://github.com/tier4/t4-devkit@v0.8.0"

python tools/t4_sam3d_align.py \
    --data-root data/t4dataset/my_scene \
    --config checkpoints/hf/pipeline.yaml \
    --camera CAM_FRONT --category car \
    --out-dir out/aligned
```

Each object is written as `NNN_<category>.ply` plus a `NNN_<category>.json`
alignment report. Add `--dry-run` to list the selected boxes and dump mask
previews without loading the model.

## Why an alignment step is needed

`output["gs"].save_ply(...)` — what `demo.py` writes — is the object in SAM 3D's
**canonical frame**: a `[-0.5, 0.5]³` cube with no metric scale and no camera
alignment (`SLatGaussianDecoder.to_representation` builds it with
`aabb=[-0.5, -0.5, -0.5, 1, 1, 1]`). The layout head separately predicts
`output["rotation"] / ["translation"] / ["scale"]`, which `make_scene()` applies
to place the object in the camera frame. That pose is metric only up to MoGe's
scale/shift normalisation, and its heading carries a few degrees of error. The T4
annotation supplies exactly what is missing.

## The four frames, and the one that bites

| Frame | Convention | Where it comes from |
| --- | --- | --- |
| `obj` | Z-up unit cube `[-0.5, 0.5]³` | `Gaussian.get_xyz` |
| `p3d` | **+X left, +Y up, +Z forward** | PyTorch3D; what SAM 3D's layout predicts into |
| `cam` | **+X right, +Y down, +Z forward** | OpenCV; T4 camera sensor frame, projects as `u = K·p` |
| `box` | **+X forward, +Y left, +Z up** | `Box3D.corners()`; `Box3D.size` is `(width, length, height)` |

`p3d` and `cam` differ by `diag(-1, -1, 1)` — a half turn about the optical axis.
SAM 3D applies it to the MoGe point map in `camera_to_pytorch3d_camera()`
(`inference_pipeline_pointmap.py`) and never undoes it, so every pose it returns
lives in `p3d`. Skipping that flip on the way back does **not** produce an obvious
error: a front camera looks along the ground, so the symptom is a car that is
upside down rather than one that is visibly misplaced. That case is pinned by
`tests/test_t4_devkit_integration.py::test_forgetting_the_camera_flip_turns_the_car_upside_down`.

Two more places to trip on:

* SAM 3D's pose uses PyTorch3D's **row-vector** convention (`p @ R`), so the
  column-vector rotation is `quaternion_to_matrix(q).T`. `make_scene()` composes
  the splat quaternions with `quaternion_invert(q)` for the same reason.
* `Box3D.size` is `(width, length, height)`, i.e. `(Y, X, Z)` in the box frame —
  not `(x, y, z)`. Reading it as XYZ swaps a car's length and width.

## What the correction does

1. **Rotation.** The residual `obj → box` rotation is snapped to the nearest of
   the 24 axis-aligned rotations (`--rotation-mode snap24`, the default). SAM 3D
   reliably gets the *discrete* pose right — which end is the front, which way is
   up — so snapping removes the residual few degrees while keeping the decision
   the model made from the image. `--rotation-mode yaw` corrects the heading only
   and preserves a predicted roll/pitch (a car on a slope); `none` keeps SAM 3D's
   rotation as-is. If SAM 3D reads a symmetric vehicle back-to-front, add
   `--extra-yaw-deg 180`.
2. **Scale.** The reconstruction is measured along the box axes (1st/99th
   percentile, so a stray splat cannot set the scale) and refitted to the box's
   `(width, length, height)`. `--scale-mode iso` (default) keeps the shape and
   takes the median of the three ratios; `length`/`width`/`height` drive it from a
   single axis; `axis` stretches each axis to fill the box exactly, at the cost of
   distorting the shape.
3. **Translation.** The reconstruction is re-centred on the box, since SAM 3D's
   translation is in MoGe units. `--z-align bottom` seats it on the box floor
   instead of centring it, which usually looks better for vehicles.
   `--keep-translation` keeps SAM 3D's offset, converted to metres.

The Gaussian covariances are carried through the same map, not just the centres:
a similarity rotates the quaternions and scales the radii, and an anisotropic
`--scale-mode axis` re-derives both from the SVD of `A·R·diag(s)`.

## Output frames

`--out-frame` picks what the exported PLY is expressed in:

* `box` (default) — box-local, so the car's nose points at **+X**. Best for
  building an asset library.
* `camera` — the T4 camera sensor frame (OpenCV).
* `base_link` — the ego frame; the object sits where the annotation says.
* `map` — world coordinates via `ego_pose`.

`--viewer-axes gltf` adds a final swap to the Y-up / −Z-forward convention most
web splat viewers assume. Without it a `base_link` or `box` export is Z-up and
will appear lying on its side.

## Masks

SAM 3D needs a mask, not a box. `--mask-source auto` (default) uses the annotated
instance mask from `object_ann` when the dataset has 2D annotations, and
otherwise fills the convex hull of the projected 3D box corners. The hull
includes background around the object, so prefer `ann` where it exists, and use
`--mask-dilate` sparingly.

## Using it from Python

```python
from sam3d_objects.integrations.t4 import dataset, pipeline

t4 = dataset.load_t4("data/t4dataset/my_scene")
frame = dataset.load_camera_frame(t4, sample_index=0, channel="CAM_FRONT")
boxes = dataset.select_boxes(frame, categories=["car"], max_distance=40.0)

result = pipeline.reconstruct_box(
    inference, t4, frame, boxes[0], out_frame="base_link", z_align="bottom"
)
result.save_ply("car.ply")
print(result.alignment.report["yaw_correction_deg"])
```

The coordinate math is `numpy` only and has no CUDA, checkpoint or dataset
dependency — see `tests/test_t4_alignment.py`.
