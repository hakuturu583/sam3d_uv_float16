#!/usr/bin/env python3
"""Fly a camera and a lidar once around an asset, through ``splatad_kernel``.

Both sensors read the same splats. The camera pass uses the colour channels; the
lidar pass uses the intensity / ray-drop / opacity that ``t4_lidar_attributes.py``
fitted, and returns the median (first) return, which is what a real spinning
sensor reports. Each frame puts the three side by side -- RGB, lidar intensity,
lidar range -- so the two modalities can be checked against each other.

    python tools/t4_lidar_turntable.py --asset out/lidar_assets/000_bus.ply \\
        --out out/orbit/000_bus.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Set before anything imports the package: `sam3d_objects.__init__` pulls in an
# internal init module that is not part of the public release.
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

# Velodyne HDL-64E S3, as splatsim configures it: 64 beams in two stacked
# 32-laser blocks, +2.0 deg down to -24.33 deg, and 2083 azimuth cells per turn
# (0.1728 deg) at 10 Hz. The blocks have different spacings -- 0.333 deg in the
# upper, 0.5 deg in the lower -- which is why the scan lines crowd towards the
# horizon and spread out below it.
HDL64E_UPPER = np.linspace(2.0, -8.33, 32)
HDL64E_LOWER = np.linspace(-8.83, -24.33, 32)
#: Beam elevations in degrees, ascending, as the kernel's tile search wants them.
HDL64E_ELEVATIONS = np.sort(np.concatenate([HDL64E_UPPER, HDL64E_LOWER]))
HDL64E_COLUMNS = 2083
HDL64E_AZIMUTH_RESOLUTION = 360.0 / HDL64E_COLUMNS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, help="ply with the lidar channels")
    parser.add_argument("--out", required=True, help="mp4 to write")
    parser.add_argument("--frames", type=int, default=120, help="frames for the full turn")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument(
        "--lidar-pose",
        choices=("level", "viewpoint", "camera"),
        default="level",
        help="where the lidar sits: 'level' stands it on the object's bearing at "
        "--lidar-height, keeping the viewpoint's parallax; 'viewpoint' shares the "
        "viewpoint's position; 'camera' shares its full pose, tilt included",
    )
    parser.add_argument(
        "--lidar-height",
        default="auto",
        help="lidar height above the object's ground plane in metres, or 'auto' "
        "(default). A beam table reaching only +2 deg above its own horizon must be "
        "mounted about as high as the roof it is meant to see; any higher and the "
        "sweep is all roof, which is what a fixed height does to a low car.",
    )
    parser.add_argument(
        "--above-roof",
        type=float,
        default=0.3,
        help="with --lidar-height auto, how far above the object's roof the sensor "
        "sits, metres. Keeping it just clear of the roof puts the whole vehicle "
        "inside the beam table without looking down on it.",
    )
    parser.add_argument(
        "--max-range", type=float, default=120.0, help="HDL-64E maximum range, metres"
    )
    parser.add_argument(
        "--view-azimuth", type=float, default=40.0, help="viewpoint bearing, degrees"
    )
    parser.add_argument(
        "--view-height", type=float, default=6.0, help="viewpoint height above ground, metres"
    )
    parser.add_argument(
        "--title", default=None, help="caption for the video (default: from the asset name)"
    )
    parser.add_argument("--device", default="cuda")
    return parser


def load_asset(path, device):
    """Splats as device tensors, falling back to colour when the lidar fit is absent."""
    from sam3d_objects.integrations.t4.asset import read_splats

    splats = read_splats(path)
    intensity = splats.intensity if splats.has_lidar else splats.colours.mean(1)
    raydrop = (
        splats.raydrop_logit if splats.has_lidar else np.full(len(splats), -4.0, np.float32)
    )
    lidar_opacity = splats.lidar_opacity if splats.has_lidar else splats.opacity

    to_device = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device)
    return dict(
        means=to_device(splats.means),
        quats=to_device(splats.quats),
        scales=to_device(splats.scales),
        colours=to_device(splats.colours),
        opacity=to_device(splats.opacity),
        intensity=to_device(intensity),
        raydrop=to_device(raydrop),
        lidar_opacity=to_device(lidar_opacity),
        has_lidar=splats.has_lidar,
    )


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """World->sensor, OpenCV convention (+X right, +Y down, +Z forward)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, float))
    right /= np.linalg.norm(right)
    upward = np.cross(right, forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.stack([right, -upward, forward])
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def sensor_frame(eye, target):
    """World->sensor for a lidar: +X forward, +Y left, +Z up, as a spinner sees it."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.stack([forward, left, up])
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def _font(size):
    """A legible face at video width; PIL's built-in bitmap font is far too small."""
    try:
        from matplotlib import font_manager
        from PIL import ImageFont

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
    except Exception:
        from PIL import ImageFont

        return ImageFont.load_default()


def _yaw(angle_rad):
    """Rotation about the object's vertical axis."""
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], np.float32)


def _spin_about(centre, angle_rad):
    """Object->world for a yaw of ``angle_rad`` about the object's own axis."""
    rotation = _yaw(angle_rad)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(centre, np.float32) - rotation @ np.asarray(centre, np.float32)
    return matrix


def _direction(azimuth_deg, elevation_deg):
    """Unit vector for a beam, in the lidar's own frame (+X forward, +Y left)."""
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def spherical_extent(points, matrix):
    """Azimuth/elevation window the object occupies from one sensor pose."""
    local = points @ matrix[:3, :3].T + matrix[:3, 3]
    distance = np.linalg.norm(local, axis=1)
    azimuth = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(local[:, 2] / np.maximum(distance, 1e-9), -1, 1)))
    return azimuth.min(), azimuth.max(), elevation.min(), elevation.max()


def colourise(values, valid, invert=False):
    """Turbo-map a scalar field, leaving the misses white."""
    import matplotlib.cm as cm

    out = np.ones((*values.shape, 3), np.float32)
    if valid.any():
        low, high = values[valid].min(), values[valid].max()
        norm = (values - low) / max(high - low, 1e-6)
        if invert:
            norm = 1.0 - norm
        out[valid] = cm.turbo(np.clip(norm[valid], 0, 1))[..., :3]
    return (out * 255).astype(np.uint8)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from splatad_kernel import lidar_rasterization
    from splatad_kernel.rendering import rasterization

    device = args.device
    asset = load_asset(args.asset, device)
    means_np = asset["means"].cpu().numpy()
    keep = asset["opacity"].cpu().numpy() > 0.05
    low, high = means_np[keep].min(0), means_np[keep].max(0)
    centre, extent = (low + high) / 2, high - low
    print(
        f"{args.asset}: {len(means_np)} splats, bbox {np.round(extent, 2)} m, "
        f"lidar channels {'present' if asset['has_lidar'] else 'MISSING (falling back to colour)'}"
    )

    focal = 0.9 * args.width
    radius = focal * float(np.hypot(extent[0], extent[1])) / (0.75 * args.width)
    # The sensor rides just above the object's roof. Fixing it in absolute terms
    # makes the sweep all roof on a low car and all flank on a bus; tying it to
    # the object keeps the whole vehicle inside the beam table either way.
    ground = float(low[2])
    roof = float(high[2]) - ground
    if str(args.lidar_height).lower() == "auto":
        lidar_height = roof + args.above_roof
        bottom = np.degrees(np.arctan2(-lidar_height, radius))
        print(
            f"lidar height {lidar_height:.2f} m = roof {roof:.2f} m + {args.above_roof:.2f} m; "
            f"at {radius:.1f} m the object spans {bottom:.1f} to "
            f"{np.degrees(np.arctan2(roof - lidar_height, radius)):.1f} deg, "
            f"beam table {HDL64E_ELEVATIONS[0]:.1f} to +{HDL64E_ELEVATIONS[-1]:.1f} deg"
        )
    else:
        lidar_height = float(args.lidar_height)
    args.lidar_height = lidar_height
    sensor_z = ground + lidar_height
    K = torch.tensor(
        [[[focal, 0, args.width / 2], [0, focal, args.height / 2], [0, 0, 1]]],
        dtype=torch.float32,
        device=device,
    )

    panel_h = args.height
    # The lidar stands off to one side; the viewer looks from a raised
    # three-quarter angle, so the sweep's lines are seen across the body
    # instead of end-on.
    viewer_eye = np.array(
        [
            centre[0] + radius * np.cos(np.radians(args.view_azimuth)),
            centre[1] + radius * np.sin(np.radians(args.view_azimuth)),
            ground + args.view_height,
        ]
    )
    viewer_view = look_at(viewer_eye, centre)

    if args.lidar_pose == "level":
        lidar_eye = np.array([centre[0] + radius, centre[1], sensor_z])
    else:
        lidar_eye = viewer_eye
    if args.lidar_pose == "camera":
        # The sensor's own axes, aimed exactly like the camera: +X along the
        # optical axis, +Y left, +Z up. A spinning lidar's beam table is
        # measured in THIS frame, so tilting the sensor tilts the beams with it.
        optical = viewer_view[:3, :3]
        lidar_view = np.eye(4, dtype=np.float32)
        lidar_view[:3, :3] = np.stack([optical[2], -optical[0], -optical[1]])
        lidar_view[:3, 3] = -lidar_view[:3, :3] @ lidar_eye
    else:
        lidar_view = sensor_frame(lidar_eye, np.array([centre[0], centre[1], lidar_eye[2]]))
    print(
        f"lidar: {args.lidar_pose} pose at {np.round(lidar_eye, 2).tolist()}, "
        f"{lidar_eye[2] - ground:.2f} m above the ground plane"
    )

    beam_elevations = HDL64E_ELEVATIONS
    rows = len(beam_elevations)
    min_el = float(beam_elevations[0] - 1.0)
    max_el = float(beam_elevations[-1] + 1.0)
    midpoints = (beam_elevations[1:] + beam_elevations[:-1]) / 2
    boundaries = torch.as_tensor(
        np.concatenate([[min_el], midpoints, [max_el]]), dtype=torch.float32, device=device
    )
    row_elevations = torch.as_tensor(beam_elevations, dtype=torch.float32, device=device)
    az_res = HDL64E_AZIMUTH_RESOLUTION
    tile_width, tile_height = 8, 1

    K = torch.tensor(
        [[[focal, 0, args.width / 2], [0, focal, panel_h / 2], [0, 0, 1]]],
        dtype=torch.float32,
        device=device,
    )

    title = args.title or (
        f"{Path(args.asset).stem}: HDL-64E lidar simulation of a SAM 3D asset"
    )
    setup_line = (
        f"64 beams  +2.0 to -24.33 deg  |  {az_res:.4f} deg azimuth ({HDL64E_COLUMNS} per turn)"
        f"  |  {args.max_range:.0f} m range  |  sensor {args.lidar_pose}, "
        f"{lidar_eye[2] - ground:.1f} m high, {radius:.1f} m from the object"
        f"  |  splatad_kernel"
    )
    title_size, small_size = 30, 19
    title_font, small_font = _font(title_size), _font(small_size)
    banner = 24 + title_size + small_size + 14
    # x264 wants even dimensions; an odd sheet height makes it drop every frame.
    sheet_height = panel_h + banner + 40
    sheet_height += sheet_height % 2
    sheet_width = args.width * 3 + 40
    sheet_width += sheet_width % 2

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps, macro_block_size=1)
    for index in range(args.frames):
        angle = 2 * np.pi * index / args.frames
        # Turning the object is the same as composing its rotation into each
        # sensor's view matrix, which spares transforming 300k splats a frame.
        spin = _spin_about(centre, angle)
        camera_view = torch.from_numpy((viewer_view @ spin).astype(np.float32)).to(device)[None]
        rgb, alpha, _ = rasterization(
            means=asset["means"],
            quats=asset["quats"],
            scales=asset["scales"],
            opacities=asset["opacity"],
            colors=asset["colours"],
            velocities=None,
            viewmats=camera_view,
            Ks=K,
            width=args.width,
            height=panel_h,
            rolling_shutter_direction=5,
            render_mode="RGB",
            rasterize_mode="antialiased",
        )
        camera_panel = ((rgb[0] + (1.0 - alpha[0])).clamp(0, 1) * 255).byte().cpu().numpy()

        # Where the object's silhouette falls in the lidar's azimuth window this
        # frame; the sweep is rendered as that sector of the full ring.
        spun_view = (lidar_view @ spin).astype(np.float32)
        object_min_az, object_max_az, _, _ = spherical_extent(means_np[keep], spun_view)
        object_min_az, object_max_az = object_min_az - 1.0, object_max_az + 1.0
        tile_span = az_res * tile_width
        tile_col_offset = int(np.floor((object_min_az + 180.0) / tile_span))
        sector_start = -180.0 + tile_col_offset * tile_span
        cols = max(int(np.ceil((object_max_az - sector_start) / tile_span)), 1) * tile_width

        raster = np.zeros((1, rows, cols, 4), np.float32)
        azimuths = sector_start + (np.arange(cols) + 0.5) * az_res
        raster[0, ..., 0] = azimuths
        raster[0, ..., 1] = beam_elevations[:, None]
        raster[0, ..., 2] = args.max_range

        features = torch.stack([asset["intensity"], asset["raydrop"]], dim=-1)[None]
        rendered, lidar_alpha, _, meta = lidar_rasterization(
            means=asset["means"],
            quats=asset["quats"],
            scales=asset["scales"],
            opacities=asset["lidar_opacity"],
            lidar_features=features,
            velocities=None,
            viewmats=torch.from_numpy(spun_view.astype(np.float32)).to(device)[None],
            raster_pts=torch.from_numpy(raster).to(device),
            tile_elevation_boundaries=boundaries,
            min_azimuth=-180.0,
            max_azimuth=180.0,
            min_elevation=min_el,
            max_elevation=max_el,
            n_elevation_channels=rows,
            azimuth_resolution=float(az_res),
            tile_width=tile_width,
            tile_height=tile_height,
            tile_col_offset=tile_col_offset,
            row_elevations=row_elevations,
            far_plane=args.max_range,
            compute_alpha_sum_until_points=False,
        )

        hit = (lidar_alpha[0, ..., 0] > 0.3).cpu().numpy()
        returned = hit & (torch.sigmoid(rendered[0, ..., 1]).cpu().numpy() < 0.5)
        intensity = rendered[0, ..., 0].cpu().numpy()
        distance = meta["median_depths"][0, ..., 0].cpu().numpy()
        returned &= distance > 0

        # Range times beam direction is the measurement itself: a point cloud in
        # the sensor's frame, which is the frame the viewer is fixed in.
        grid_az, grid_el = np.meshgrid(azimuths, beam_elevations, indexing="xy")
        points = distance[..., None] * np.stack(
            [
                np.cos(np.radians(grid_el)) * np.cos(np.radians(grid_az)),
                np.cos(np.radians(grid_el)) * np.sin(np.radians(grid_az)),
                np.sin(np.radians(grid_el)),
            ],
            axis=-1,
        )
        world = points @ lidar_view[:3, :3] - lidar_view[:3, :3].T @ lidar_view[:3, 3]
        seen = world @ viewer_view[:3, :3].T + viewer_view[:3, 3]

        def cloud(colours):
            """Draw the sweep's points from the viewer's fixed viewpoint."""
            canvas = np.full((panel_h, args.width, 3), 255, np.uint8)
            visible = returned & (seen[..., 2] > 0.05)
            if visible.any():
                u = focal * seen[..., 0] / seen[..., 2] + args.width / 2
                v = focal * seen[..., 1] / seen[..., 2] + panel_h / 2
                inside = (
                    visible & (u >= 0) & (u < args.width - 1) & (v >= 0) & (v < panel_h - 1)
                )
                rr, cc = v[inside].astype(int), u[inside].astype(int)
                # Nearest point wins the pixel, so the far side does not show
                # through the near one.
                order = np.argsort(-seen[..., 2][inside])
                for dr in (0, 1):
                    for dc in (0, 1):
                        canvas[
                            np.clip(rr[order] + dr, 0, panel_h - 1),
                            np.clip(cc[order] + dc, 0, args.width - 1),
                        ] = colours[inside][order]
            return Image.fromarray(canvas)

        intensity_panel = cloud(colourise(intensity, returned))
        range_panel = cloud(colourise(distance, returned, invert=True))

        sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((14, 10), title, fill=(20, 20, 20), font=title_font)
        draw.text((14, 16 + title_size), setup_line, fill=(90, 90, 90), font=small_font)
        draw.text(
            (14, 20 + title_size + small_size),
            f"object yaw {np.degrees(angle):6.1f} deg   |   {int(returned.sum()):6d} returns "
            f"this sweep",
            fill=(90, 90, 90),
            font=small_font,
        )
        for position, (label, image) in enumerate(
            (
                ("CAMERA  (rendered from the splats)", Image.fromarray(camera_panel)),
                ("HDL-64E RETURNS  coloured by intensity", intensity_panel),
                ("HDL-64E RETURNS  coloured by range", range_panel),
            )
        ):
            x = 10 + position * (args.width + 10)
            sheet.paste(image, (x, banner + 24))
            draw.text((x + 2, banner + 4), label, fill=(20, 20, 20), font=small_font)
        writer.append_data(np.asarray(sheet))
        if index % 20 == 0:
            print(f"  frame {index + 1}/{args.frames}: {int(returned.sum())} returns", flush=True)

    writer.close()
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
