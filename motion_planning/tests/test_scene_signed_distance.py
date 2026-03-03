#!/usr/bin/env python3
"""Regression tests for scene signed-distance behavior."""

from __future__ import annotations

import numpy as np

from motion_planning.geometry.scene import Scene


def test_point_signed_distance_obb_is_metric() -> None:
    scene = Scene()
    scene.add_block(size=(1.0, 1.0, 1.0), position=(0.0, 0.0, 0.0))

    d_center = scene.signed_distance(np.array([0.0, 0.0, 0.0], dtype=float))
    d_surface = scene.signed_distance(np.array([0.5, 0.0, 0.0], dtype=float))
    d_far = scene.signed_distance(np.array([2.0, 0.0, 0.0], dtype=float))

    assert np.isclose(d_center, -0.5, atol=1e-6), d_center
    assert abs(d_surface) <= 1e-6, d_surface
    assert np.isclose(d_far, 1.5, atol=1e-6), d_far


def test_block_signed_distance_no_false_minus_one_for_small_overlap() -> None:
    scene = Scene()
    scene.add_block(size=(1.0, 1.0, 1.0), position=(0.0, 0.0, 0.0))
    # Small overlap along +x (penetration about 0.01m).
    d = scene.signed_distance_block(
        size=(1.0, 1.0, 1.0),
        position=np.array([0.99, 0.0, 0.0], dtype=float),
        quat=(0.0, 0.0, 0.0, 1.0),
    )
    assert d < 0.0, d
    assert d > -0.2, d


if __name__ == "__main__":
    test_point_signed_distance_obb_is_metric()
    test_block_signed_distance_no_false_minus_one_for_small_overlap()
    print("scene signed-distance tests passed")

