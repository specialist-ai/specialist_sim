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


def _scene(embodiment: EmbodimentCfg) -> Scene:
  return Scene(SceneCfg(), embodiment.build())


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
