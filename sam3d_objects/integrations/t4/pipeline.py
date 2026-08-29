"""End-to-end: T4 camera frame -> SAM 3D reconstruction -> box-aligned splats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .align import BoxAlignment, align_to_box, check_choice, compose_alignment
from .dataset import CameraFrame, box_mask
from .frames import FRAME_CHAIN, VIEWER_AXES
from .gaussian_ops import opaque_positions, transform_gaussian

__all__ = ["ObjectResult", "align_output_to_box", "reconstruct_box"]


@dataclass
class ObjectResult:
    """One reconstructed and box-aligned object."""

    instance_token: str
    category: str
    alignment: BoxAlignment
    gaussian: Any

    @property
    def frame(self) -> str:
        """Frame the splats are expressed in."""
        return self.alignment.frame

    def save_ply(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.gaussian.save_ply(str(path))

    def save_report(self, path) -> None:
        payload = {
            "instance_token": self.instance_token,
            "category": self.category,
            "frame": self.frame,
            "linear": self.alignment.linear.tolist(),
            "translation": self.alignment.translation.tolist(),
            **self.alignment.report,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2))


def align_output_to_box(
    output: dict,
    frame: CameraFrame,
    box,
    *,
    out_frame: str = "box",
    viewer_axes: str | None = None,
    **align_kwargs,
) -> BoxAlignment:
    """Solve the alignment for one SAM 3D ``output`` and its T4 box.

    Args:
        output: The dict returned by ``Inference.__call__``. Needs ``"gs"``,
            ``"rotation"``, ``"translation"`` and ``"scale"``.
        frame: The :class:`~sam3d_objects.integrations.t4.dataset.CameraFrame`
            the image came from.
        box: The ``Box3D`` (camera frame) the object was masked from.
        out_frame: One of :data:`~sam3d_objects.integrations.t4.frames.FRAME_CHAIN`.
        viewer_axes: Key into
            :data:`~sam3d_objects.integrations.t4.frames.VIEWER_AXES`, applied
            last. ``"gltf"`` produces Y-up splats for web viewers; ``None``
            leaves the frame's own axes alone.
        **align_kwargs: Forwarded to
            :func:`~sam3d_objects.integrations.t4.align.align_to_box`.
    """
    check_choice("out_frame", out_frame, FRAME_CHAIN)
    if viewer_axes is not None:
        check_choice("viewer_axes", viewer_axes, VIEWER_AXES)

    gaussian = output["gs"]
    obj_points = opaque_positions(gaussian).detach().cpu().float().numpy()
    rotation, translation, scale = _layout_pose(output)

    alignment = align_to_box(
        obj_points,
        sam3d_rotation=rotation,
        sam3d_translation=translation,
        sam3d_scale=scale,
        box_rotation=box.rotation.rotation_matrix,
        box_position=np.asarray(box.position, dtype=np.float64),
        box_size=np.asarray(box.size, dtype=np.float64),
        **align_kwargs,
    )

    # The frames form one linear chain, so the hops to walk are exactly the
    # prefix of FRAME_CHAIN up to the one asked for.
    hops = (
        ("camera", box.rotation.rotation_matrix, np.asarray(box.position, dtype=np.float64)),
        ("base_link", frame.rot_ego_cam, frame.trans_ego_cam),
        ("map", frame.rot_map_ego, frame.trans_map_ego),
    )
    for name, hop_rotation, hop_translation in hops[: FRAME_CHAIN.index(out_frame)]:
        alignment = compose_alignment(alignment, hop_rotation, hop_translation, name)

    if viewer_axes is not None:
        alignment = compose_alignment(
            alignment, VIEWER_AXES[viewer_axes], np.zeros(3), f"{alignment.frame}+{viewer_axes}"
        )
    return alignment


def reconstruct_box(
    inference,
    t4,
    frame: CameraFrame,
    box,
    *,
    seed: int | None = 42,
    mask_source: str = "auto",
    mask_dilate: int = 0,
    mask_dir=None,
    out_frame: str = "box",
    viewer_axes: str | None = None,
    **align_kwargs,
) -> ObjectResult | None:
    """Reconstruct one annotated object and align it to its box.

    Args:
        inference: A ``notebook.inference.Inference`` instance.
        t4: The open ``T4Devkit``.
        frame: The camera frame to read from.
        box: One of ``frame.boxes``.
        seed: Diffusion seed.
        mask_source: See :func:`~sam3d_objects.integrations.t4.dataset.box_mask`.
        mask_dilate: Mask dilation in pixels.
        mask_dir: Directory of precomputed masks, for ``mask_source="file"``.
        out_frame: Export frame, one of
            :data:`~sam3d_objects.integrations.t4.frames.FRAME_CHAIN`.
        viewer_axes: Optional final axis swap for third party viewers.
        **align_kwargs: Forwarded to
            :func:`~sam3d_objects.integrations.t4.align.align_to_box`.

    Returns:
        The aligned result, or ``None`` when no usable mask could be built.
    """
    mask = box_mask(t4, frame, box, source=mask_source, dilate=mask_dilate, mask_dir=mask_dir)
    if mask is None or not mask.any():
        return None

    output = inference(frame.image, mask, seed=seed)
    alignment = align_output_to_box(
        output,
        frame,
        box,
        out_frame=out_frame,
        viewer_axes=viewer_axes,
        **align_kwargs,
    )
    gaussian = transform_gaussian(output["gs"], alignment.linear, alignment.translation)
    return ObjectResult(
        instance_token=box.uuid or "",
        category=box.semantic_label.name,
        alignment=alignment,
        gaussian=gaussian,
    )


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _layout_pose(output: dict):
    """Pull the decoded layout out of a SAM 3D output, checking it *is* decoded.

    When the pipeline is configured with the ``"default"`` pose decoder (see
    ``InferencePipeline.init_pose_decoder``) the layout head's raw activations
    stay in the output dict under the same key names, so a silent misalignment is
    easy to walk into. The decoded scale is an exponential and therefore strictly
    positive; the raw one is a log and is routinely negative.
    """
    missing = [key for key in ("rotation", "translation", "scale") if key not in output]
    if missing:
        raise KeyError(
            f"SAM 3D output has no {missing} -- this pipeline predicts no object layout, "
            "so there is nothing to align a T4 box against. Check "
            "`pose_target_convention` in the sparse-structure generator config."
        )

    rotation = _to_numpy(output["rotation"])[:4]
    translation = _to_numpy(output["translation"])[:3]
    scale = _to_numpy(output["scale"])[:3]
    if np.any(scale <= 0.0):
        raise ValueError(
            f"SAM 3D layout scale {scale} is not positive, which means the pose was never "
            "decoded (pose_decoder_name='default' leaves the raw log-scale in place). "
            "Configure a real convention such as 'ScaleShiftInvariant'."
        )
    return rotation, translation, scale
