"""桌面、底座与原有碰撞几何的场景回归。"""

import mujoco
import numpy as np
import pytest

from so101.collision_scene import inspect_contacts, load_collision_scene


SCENE = "configs/so101/scene_table_multi.json"


def test_table_and_base_preserve_arm_collision_geometry():
    model, _, _ = load_collision_scene(scene_path=SCENE)
    table = model.geom("table")
    assert table.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    assert np.isclose(table.pos[2] + table.size[2], -0.004)
    for body_name in (
        "base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1",
    ):
        body_id = model.body(body_name).id
        assert np.any(
            (model.geom_bodyid == body_id)
            & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        ), body_name
    assert model.geom("practice_sphere").id >= 0
    assert model.geom("obstacle_box").id >= 0


@pytest.mark.parametrize(
    "degrees, category",
    [
        ([60.2703, -12.2243, 69.446, 37.4999, -127.0741], "table"),
        ([-28.4244, 85.353, 27.8609, 61.3247, -15.3183], "base_collision"),
        ([56.7793, -29.0948, 91.1553, 74.693, 91.8718], "self"),
    ],
)
def test_scene_detects_environment_and_self_contacts(degrees, category):
    model, config, joint_ids = load_collision_scene(False, SCENE)
    _, contacts = inspect_contacts(model, config, joint_ids, np.deg2rad(degrees))
    penetrating = [contact for contact in contacts if contact["dist"] < 0]
    if category == "self":
        assert any(
            "table" not in (contact["geom1"], contact["geom2"])
            and "base_collision" not in (contact["geom1"], contact["geom2"])
            for contact in penetrating
        )
    else:
        assert any(category in (contact["geom1"], contact["geom2"]) for contact in penetrating)


def test_obstacle_switch_keeps_table_and_base():
    model, config, joint_ids = load_collision_scene(False, SCENE)
    assert model.geom("table").id >= 0
    assert model.geom("base_collision").id >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "practice_sphere") == -1
    _, contacts = inspect_contacts(model, config, joint_ids, np.zeros(5))
    assert not contacts


def test_legacy_sphere_scene_still_loads():
    model, config, joint_ids = load_collision_scene()
    _, contacts = inspect_contacts(model, config, joint_ids, np.zeros(5))
    assert any("practice_sphere" in (contact["geom1"], contact["geom2"]) for contact in contacts)


def test_query_margin_reports_separated_surfaces():
    angles = np.deg2rad([-38, 0, 0, 0, 0])
    model, config, joint_ids = load_collision_scene(query_margin=0.005)
    _, contacts = inspect_contacts(model, config, joint_ids, angles)
    sphere_contacts = [
        contact for contact in contacts
        if "practice_sphere" in (contact["geom1"], contact["geom2"])
    ]
    assert sphere_contacts
    assert 0 < min(contact["dist"] for contact in sphere_contacts) < 0.005
    model, config, joint_ids = load_collision_scene()
    _, contacts = inspect_contacts(model, config, joint_ids, angles)
    assert not contacts
