"""The room and what stands the hands in it, without an environment around them."""

import mujoco
import numpy as np
import pytest

from gantry import GantryCfg
from specialist_sim.embodiment import (
  ASSETS,
  ChainHand,
  EmbodimentCfg,
  FloatingCfg,
  FreeHand,
  bind_fingers,
  carried,
  check_slots,
  gripper_spec,
)
from specialist_sim.scene import Scene, SceneCfg
from specialist_sim.spaces import GRIPPERS, PER_HAND

CARRIERS = [FloatingCfg(), GantryCfg()]
IDS = ["floating", "gantry"]

TAG_PATTERNS = {
  0: (
    "..#....##",
    "#########",
    ".#.....##",
    "##.....#.",
    "##.##..##",
    "##.#.#.#.",
    ".#.....##",
    "#########",
    "..#.##.##",
  ),
  1: (
    "..#....#.",
    "#########",
    ".#.....##",
    "##.....##",
    ".#.#...#.",
    ".#..#..##",
    "##.....##",
    ".#######.",
    "#.#...##.",
  ),
}
"""The reference tagStandard41h12 images for IDs 0 and 1, top to bottom."""


def _scene(embodiment: EmbodimentCfg, *, apriltags: bool = False) -> Scene:
  return Scene(SceneCfg(apriltags=apriltags), embodiment.build())


@pytest.mark.parametrize("embodiment", CARRIERS, ids=IDS)
def test_the_room_stands_up_the_hands_it_was_given(embodiment: EmbodimentCfg) -> None:
  scene = _scene(embodiment)
  assert scene.slots == GRIPPERS
  assert len(scene.hands) == len(GRIPPERS)
  for slot, hand in zip(GRIPPERS, scene.hands, strict=True):
    assert hand.slot == slot
    assert mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, hand.base) == (
      f"{slot}_base"
    )
  assert scene.cube.rest > scene.table_top
  assert len(scene.pads) > 0, "a grasp needs pads to touch with"


def test_the_room_matches_the_measured_work_area() -> None:
  scene = _scene(FloatingCfg(slots=("right_gripper",)))
  model = scene.model
  assert model.body("table").pos == pytest.approx([0.70, 0.0, -0.285])
  assert 2.0 * model.geom("table").size == pytest.approx([0.60, 1.20, 0.05])
  assert scene.table_top == pytest.approx(-0.26)
  assert scene.tape == pytest.approx((0.70, 0.0))
  assert 2.0 * model.geom("tape").size[:2] == pytest.approx([0.01, 0.01])

  ground = model.body("ground").pos[2] + model.geom("ground").size[2]
  left = model.body("wall_left").pos[1] - model.geom("wall_left").size[1]
  far = model.body("wall_far").pos[0] - model.geom("wall_far").size[0]
  assert ground == pytest.approx(-1.0)
  assert left == pytest.approx(0.80)
  assert far == pytest.approx(1.20)
  for name in ("ground", "wall_left", "wall_far"):
    assert model.geom(name).contype == 0 and model.geom(name).conaffinity == 0


def test_the_room_uses_textures_on_its_large_surfaces() -> None:
  model = _scene(FloatingCfg(slots=("right_gripper",))).model
  for name in ("ground", "wall_left", "wall_far", "table"):
    material = int(model.geom_matid[model.geom(name).id])
    assert material >= 0
    texture = model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB]
    assert texture >= 0
    assert model.tex_type[texture] == mujoco.mjtTexture.mjTEXTURE_2D


def test_apriltags_are_an_opt_in_room_addition() -> None:
  without = _scene(FloatingCfg(slots=("right_gripper",))).model
  with_tags = _scene(FloatingCfg(slots=("right_gripper",)), apriltags=True).model
  for tag in (0, 1):
    assert mujoco.mj_name2id(without, mujoco.mjtObj.mjOBJ_BODY, f"tag_{tag}") == -1
    assert mujoco.mj_name2id(with_tags, mujoco.mjtObj.mjOBJ_BODY, f"tag_{tag}") >= 0


def test_the_anchor_tags_hold_the_measured_poses() -> None:
  scene = _scene(FloatingCfg(slots=("right_gripper",)), apriltags=True)
  model, data = scene.model, scene.data
  mujoco.mj_forward(model, data)
  for tag, (position, facing) in {
    0: ((0.0, 0.5633, 0.0), (1.0, -1.0, 0.0)),
    1: ((0.0, -0.5633, 0.0), (1.0, 1.0, 0.0)),
  }.items():
    body = model.body(f"tag_{tag}")
    assert body.pos == pytest.approx(position)
    out = data.xmat[body.id].reshape(3, 3)[:, 2]
    assert out == pytest.approx(np.asarray(facing) / np.sqrt(2.0), abs=1e-9)


@pytest.mark.parametrize("tag", [0, 1])
def test_the_anchor_tag_ink_is_the_reference_pattern(tag: int) -> None:
  scene = _scene(FloatingCfg(slots=("right_gripper",)), apriltags=True)
  model = scene.model
  ink = [
    geom
    for geom in range(model.ngeom)
    if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "").startswith(
      f"tag_{tag}_ink_"
    )
  ]
  cell = 0.10 / 9.0
  observed = []
  for row in range(9):
    line = ""
    for column in range(9):
      point = np.array([(column - 4) * cell, (4 - row) * cell])
      black = any(
        np.all(np.abs(point - model.geom_pos[geom, :2]) <= model.geom_size[geom, :2])
        for geom in ink
      )
      line += "#" if black else "."
    observed.append(line)
  assert tuple(observed) == TAG_PATTERNS[tag]


def test_a_rig_of_one_hand_holds_only_that_slot() -> None:
  scene = _scene(FloatingCfg(slots=("right_gripper",)))
  assert scene.slots == ("right_gripper",)
  assert len(scene.hands) == 1
  assert scene.observe(scene.data.qpos[None]).shape == (1, 1, PER_HAND)


@pytest.mark.parametrize("embodiment", CARRIERS, ids=IDS)
def test_a_hand_is_read_back_where_its_carrier_put_it(
  embodiment: EmbodimentCfg,
) -> None:
  """A pose is the gripper base's, whatever holds it up."""
  scene = _scene(embodiment)
  data = mujoco.MjData(scene.model)
  mujoco.mj_resetData(scene.model, data)
  spot = np.array([0.47, 0.16, scene.table_top + 0.18])
  upright = np.array([1.0, 0.0, 0.0, 0.0])
  scene.embodiment.place(data, scene.hands[0], spot, upright)
  read = scene.observe(data.qpos[None])[0, 0]
  assert read[:3] == pytest.approx(spot, abs=1e-6)
  assert np.abs(read[3:7]) == pytest.approx(np.abs(upright), abs=1e-6)


@pytest.mark.parametrize("embodiment", CARRIERS, ids=IDS)
def test_a_hand_that_is_not_moving_has_no_twist(embodiment: EmbodimentCfg) -> None:
  scene = _scene(embodiment)
  qpos = scene.data.qpos[None]
  twists = scene.twists(qpos, np.zeros((1, scene.model.nv)))
  assert twists.shape == (1, len(scene.hands), 6)
  assert twists == pytest.approx(np.zeros_like(twists), abs=1e-9)


def test_a_free_hand_hangs_on_a_joint_a_wrench_can_pull() -> None:
  scene = _scene(FloatingCfg())
  for hand in scene.hands:
    free = carried(hand, FreeHand)
    assert free.qpos >= 0 and free.rig >= 0
  with pytest.raises(TypeError, match="is not a ChainHand"):
    carried(scene.hands[0], ChainHand)


def test_the_hands_are_named_in_the_order_the_rig_holds_them() -> None:
  assert check_slots(GRIPPERS) == GRIPPERS
  assert check_slots(["right_gripper"]) == ("right_gripper",)
  for wrong in ((), ("right_gripper", "left_gripper"), ("third_gripper",)):
    with pytest.raises(ValueError, match="in that order"):
      check_slots(wrong)


def test_a_gripper_whose_fingers_travel_differently_is_refused() -> None:
  """An opening names the finger joint's own travel, and the model must agree."""
  room = mujoco.MjSpec.from_file(str(ASSETS / "room.xml"))
  gripper = gripper_spec()
  gripper.joint("left_finger").range = [-1.0, 1.0]
  room.attach(gripper, prefix="left_gripper_", frame=room.worldbody.add_frame())
  model = room.compile()
  with pytest.raises(ValueError, match="travels"):
    bind_fingers(model, "left_gripper")
