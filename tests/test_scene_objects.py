"""Tests for the scene-object registry (perception/scene_objects.py)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

SCENE = "rex_assets/rex_simulation/pick_and_place_scene.xml"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(SCENE)


def test_discovers_all_obj_bodies(model):
    from rex_tendon.perception.scene_objects import discover_objects

    objects = discover_objects(model)
    names = {o.name for o in objects}
    assert {"obj_cube", "obj_cylinder", "obj_bar",
            "obj_cube_purple", "obj_cube_yellow"} <= names
    assert all(o.body_id >= 0 and o.geom_id >= 0 for o in objects)


def test_colors_from_names_materials_and_rgba(model):
    from rex_tendon.perception.scene_objects import discover_objects

    by_name = {o.name: o for o in discover_objects(model)}
    assert by_name["obj_cube"].color == "red"  # via material obj_red
    assert by_name["obj_cylinder"].color == "blue"  # via material obj_blue
    assert by_name["obj_bar"].color == "green"  # via material obj_green
    assert by_name["obj_cube_purple"].color == "purple"  # name hint
    assert by_name["obj_cube_yellow"].color == "yellow"  # name hint
    assert by_name["obj_cube_extra_1"].color == "orange"  # raw rgba 1,0.5,0


def test_shapes_and_sizes(model):
    from rex_tendon.perception.scene_objects import discover_objects

    by_name = {o.name: o for o in discover_objects(model)}
    assert by_name["obj_cube"].shape == "cube"
    assert by_name["obj_cylinder"].shape == "cylinder"
    assert by_name["obj_bar"].shape == "bar"
    assert by_name["obj_cube"].half_height == pytest.approx(0.01)


def test_find_by_color_and_available_colors(model):
    from rex_tendon.perception.scene_objects import (
        available_colors,
        discover_objects,
        find_by_color,
    )

    objects = discover_objects(model)
    assert find_by_color(objects, "red").name == "obj_cube"
    assert find_by_color(objects, "red", shape="cube").name == "obj_cube"
    assert find_by_color(objects, "blue", shape="cylinder").name == "obj_cylinder"
    assert find_by_color(objects, "red", shape="cylinder") is None
    assert {"red", "green", "blue", "purple", "yellow"} <= available_colors(objects)


def test_nearest_object(model):
    from rex_tendon.perception.scene_objects import discover_objects, nearest_object

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    objects = discover_objects(model)
    # Probe right at obj_cube's spawn position (0.06, -0.05).
    obj, dist = nearest_object(objects, data, np.array([0.06, -0.05, 0.012]))
    assert obj.name == "obj_cube"
    assert dist < 0.01
