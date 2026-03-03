#!/usr/bin/env python3
"""Smoke tests for the public API."""

from __future__ import annotations

from motion_planning import Scene, WorldModel, plan_path as plan


def test_scene_alias() -> None:
    """Scene is the user-facing alias for WorldModel."""
    assert Scene is WorldModel


def test_world_model_crud() -> None:
    wm = WorldModel()

    # add
    oid = wm.add_block(size=(0.4, 0.4, 0.4), position=(2.0, 0.0, 0.0), object_id="obs_0")
    b = wm.query_block(oid)
    assert b.object_id == "obs_0"
    assert b.size == (0.4, 0.4, 0.4)
    assert b.position == (2.0, 0.0, 0.0)

    # update
    wm.update_block(oid, position=(3.0, 0.0, 0.0))
    b2 = wm.query_block(oid)
    assert b2.position == (3.0, 0.0, 0.0)
    assert b2.size == (0.4, 0.4, 0.4)  # unchanged

    # remove
    wm.remove_block(oid)
    try:
        wm.query_block(oid)
        assert False, "Expected KeyError after removal"
    except (KeyError, IndexError):
        pass

    # reset
    wm.add_block(size=(0.2, 0.2, 0.2), position=(1.0, 0.0, 0.0), object_id="obs_1")
    wm.reset()
    try:
        wm.query_block("obs_1")
        assert False, "Expected KeyError after reset"
    except (KeyError, IndexError):
        pass


def test_geometric_plan_smoke() -> None:
    wm = WorldModel()
    wm.add_block(size=(0.4, 0.4, 0.4), position=(2.0, 0.0, 0.0), object_id="obs_0")

    res = plan(
        start=(0.0, 0.0, 0.0),
        end=(0.8, 0.0, 0.0),
        method="Powell",
        world_model=wm,
        moving_block_size=(0.1, 0.1, 0.1),
        config={"n_vias": 2, "n_samples_curve": 41, "safety_margin": 0.0},
        options={"maxiter": 8, "xtol": 1e-2, "ftol": 1e-2},
    )
    assert res.path.sample(11).shape == (11, 3)


if __name__ == "__main__":
    test_scene_alias()
    test_world_model_crud()
    test_geometric_plan_smoke()
    print("public api smoke test passed")

