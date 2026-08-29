"""View-selection tests: one answer to "which frames show this object cleanly".

Run on CPU with fakes; the real scan needs a dataset and `t4-devkit`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

from sam3d_objects.integrations.t4.views import (
    ObjectView,
    best_per_instance,
    fully_inside,
    projected_corners,
)


class FakeFrame:
    """Just the projection a view test needs."""

    def __init__(self, size=(1600, 900), focal=1000.0):
        self.size = size
        self.intrinsic = np.array(
            [[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]], float
        )
        self.distortion = None


class FakeBox:
    def __init__(self, corners):
        self._corners = np.asarray(corners, float)

    def corners(self):
        return self._corners


def box_corners(centre, half=(1.0, 0.8, 0.7)):
    """Eight corners of an axis-aligned box in camera coordinates."""
    centre = np.asarray(centre, float)
    offsets = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return centre + offsets * np.asarray(half, float)


def test_corners_project_to_pixels():
    frame = FakeFrame()
    uv = projected_corners(frame, FakeBox(box_corners([0, 0, 12])))
    assert uv is not None and uv.shape == (8, 2)
    # A box centred on the optical axis straddles the principal point.
    assert uv[:, 0].min() < 800 < uv[:, 0].max()
    assert uv[:, 1].min() < 450 < uv[:, 1].max()


def test_a_box_behind_the_camera_has_no_projection():
    frame = FakeFrame()
    assert projected_corners(frame, FakeBox(box_corners([0, 0, -12]))) is None


def test_a_box_straddling_the_camera_plane_has_no_projection():
    """Half in front, half behind is exactly where naive projection explodes."""
    frame = FakeFrame()
    assert projected_corners(frame, FakeBox(box_corners([0, 0, 0.2]))) is None


def test_fully_inside_respects_the_margin():
    uv = np.array([[30.0, 30.0], [1570.0, 870.0]])
    assert fully_inside(uv, (1600, 900), margin=25)
    assert not fully_inside(uv, (1600, 900), margin=40)


def test_an_object_at_the_frame_edge_is_not_whole():
    frame = FakeFrame()
    uv = projected_corners(frame, FakeBox(box_corners([9.0, 0, 12])))
    assert uv is not None
    assert not fully_inside(uv, frame.size, margin=25)


def view(instance, area, sample=0, camera="CAM_FRONT"):
    return ObjectView(
        sample_index=sample,
        sample_token=f"s{sample}",
        camera=camera,
        instance_token=instance,
        category="car",
        area_px=area,
        distance_m=10.0,
        num_lidar_pts=100,
    )


def test_best_per_instance_takes_the_largest_view_of_each():
    views = [
        view("a", 100.0, sample=0),
        view("a", 900.0, sample=5),
        view("b", 400.0, sample=2),
    ]
    best = best_per_instance(views)
    assert [v.instance_token for v in best] == ["a", "b"]  # ordered by area
    assert best[0].area_px == 900.0
    assert best[0].sample_index == 5


def test_object_view_round_trips_through_json():
    original = view("token", 123.0, sample=7, camera="CAM_BACK")
    assert ObjectView.from_dict(original.to_dict()) == original


def scored(area=1e6, aspect=45.0, lidar=1000, instance="a", sample=0):
    return ObjectView(
        sample_index=sample,
        sample_token=f"s{sample}",
        camera="CAM_FRONT",
        instance_token=instance,
        category="car",
        area_px=area,
        distance_m=10.0,
        num_lidar_pts=lidar,
        aspect_deg=aspect,
    )


def test_a_three_quarter_view_beats_a_nose_on_one_of_the_same_size():
    """Two faces pin length and width; one face leaves the rest to be guessed."""
    from sam3d_objects.integrations.t4.views import view_score

    reference = 1e6
    quarter = view_score(scored(aspect=45.0), reference_area=reference)
    nose_on = view_score(scored(aspect=2.0), reference_area=reference)
    broadside = view_score(scored(aspect=88.0), reference_area=reference)
    assert quarter > broadside > 0
    assert quarter > nose_on > 0


def test_size_still_dominates_a_marginally_better_angle():
    from sam3d_objects.integrations.t4.views import view_score

    reference = 1e6
    big = view_score(scored(area=1e6, aspect=20.0), reference_area=reference)
    tiny = view_score(scored(area=2e4, aspect=45.0), reference_area=reference)
    assert big > tiny


def test_lidar_support_breaks_a_tie():
    from sam3d_objects.integrations.t4.views import view_score

    reference = 1e6
    rich = view_score(scored(lidar=4000), reference_area=reference)
    poor = view_score(scored(lidar=5), reference_area=reference)
    assert rich > poor


def test_best_per_instance_ranks_by_score_not_area():
    """The largest frame is not automatically the one to reconstruct from."""
    from sam3d_objects.integrations.t4.views import best_per_instance

    views = [
        scored(area=1.0e6, aspect=1.0, lidar=50, sample=1),   # biggest, nose-on
        scored(area=0.9e6, aspect=45.0, lidar=3000, sample=2),  # slightly smaller, better
    ]
    assert best_per_instance(views)[0].sample_index == 2


def test_aspect_survives_the_json_round_trip():
    original = scored(aspect=33.0)
    assert ObjectView.from_dict(original.to_dict()).aspect_deg == pytest.approx(33.0)


def test_an_old_targets_file_without_aspect_still_loads():
    """Files written before the score existed must not break the pipeline."""
    payload = scored().to_dict()
    payload.pop("aspect_deg")
    assert ObjectView.from_dict(payload).aspect_deg == pytest.approx(45.0)
