"""An embodiment names the hands the room holds; the controller answers to it."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from gantry import GantryCfg
from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.action import ActionCfg, hold
from specialist_sim.calibration import CameraCfg
from specialist_sim.control import CTRL, XFRC
from specialist_sim.embodiment import EmbodimentCfg, FloatingCfg, FreeHand
from specialist_sim.observation import flat_row
from specialist_sim.render import seat
from specialist_sim.scene import Scene, SceneCfg
from specialist_sim.spaces import GRIPPERS, HORIZON, PER_HAND
from specialist_sim.task import BIMANUAL

SMALL = PickPlaceCfg(image_size=64, max_steps=10**6)
ONE = FloatingCfg(slots=(GRIPPERS[1],))
ALONE = replace(BIMANUAL, driven=ONE.slots)


def test_a_rig_of_one_hand_holds_and_shows_that_hand_alone() -> None:
  env = PickPlaceEnv(replace(SMALL, embodiment=ONE, task=ALONE))
  obs, _ = env.reset(seed=0)
  assert set(obs) == {"row", "right_gripper"}
  assert set(env.cameras) == {"right_gripper"}
  assert [hand.slot for hand in env.scene.hands] == ["right_gripper"]
  assert env.action_space.shape == (HORIZON, PER_HAND)
  assert flat_row(obs).shape == (PER_HAND,)
  lens = env.scene.model.body("right_gripper_lens").id
  assert seat(env.scene.model, "right_gripper") == lens
  with pytest.raises(ValueError, match="names 1 hands"):
    env.reset(seed=0, options={"state": np.zeros(2 * PER_HAND)})
  plan = hold(flat_row(obs))
  plan[:, 0] += 0.01
  obs, *_ = env.step(plan)
  assert flat_row(obs)[0] > 0.0


def test_the_rig_the_task_and_the_cameras_name_the_same_hands() -> None:
  with pytest.raises(ValueError, match="the room holds"):
    PickPlaceEnv(replace(SMALL, embodiment=ONE, task=BIMANUAL))
  with pytest.raises(ValueError, match="in that order"):
    FloatingCfg(slots=tuple(reversed(GRIPPERS))).build()
  with pytest.raises(ValueError, match="the cameras are"):
    both = dict.fromkeys(GRIPPERS, CameraCfg())
    PickPlaceEnv(replace(SMALL, embodiment=ONE, task=ALONE, cameras=both))


def test_the_impedance_pulls_on_free_rigs_only() -> None:
  floating = PickPlaceEnv(SMALL)
  assert all(isinstance(hand, FreeHand) for hand in floating.scene.hands)
  assert floating.controller.drives == CTRL | XFRC
  gantry = PickPlaceEnv(replace(SMALL, embodiment=GantryCfg()))
  assert not any(isinstance(hand, FreeHand) for hand in gantry.scene.hands)
  assert gantry.controller.drives == CTRL
  with pytest.raises(TypeError, match="not a FreeHand"):
    PickPlaceEnv(replace(SMALL, embodiment=GantryCfg(), controller=ActionCfg()))


def test_a_carrier_reports_its_base_where_it_stands(embodiment: EmbodimentCfg) -> None:
  env = PickPlaceEnv(replace(SMALL, embodiment=embodiment))
  obs, _ = env.reset(seed=5)
  base, data = env.scene.hands[0].base, env.datas[0]
  assert flat_row(obs)[:3] == pytest.approx(data.xpos[base], abs=1e-6)
  assert flat_row(obs)[3:7] == pytest.approx(data.xquat[base], abs=1e-6)


def test_a_hand_reports_the_twist_its_joint_carries_it_with(
  embodiment: EmbodimentCfg,
) -> None:
  """A hand's velocity is read the same way whatever carries it: a free hand's
  is its own joint's, and a carried hand's follows its chain."""
  scene = Scene(SceneCfg(), embodiment.build())
  data = mujoco.MjData(scene.model)
  rng = np.random.default_rng(0)
  data.qvel[:] = rng.normal(size=len(data.qvel)) * 0.1
  twists = scene.twists(data.qpos[None].copy(), data.qvel[None].copy())
  assert twists.shape == (1, len(scene.hands), 6)
  assert np.isfinite(twists).all()
  mujoco.mj_kinematics(scene.model, data)
  for index, hand in enumerate(scene.hands):
    if not isinstance(hand, FreeHand):
      continue
    dof = scene.model.jnt_dofadr[scene.model.body_jntadr[hand.rig]]
    spin = data.xmat[hand.rig].reshape(3, 3) @ data.qvel[dof + 3 : dof + 6]
    want = np.concatenate([data.qvel[dof : dof + 3], spin])
    assert np.allclose(twists[0, index], want, atol=1e-12)


def test_a_hand_at_rest_has_no_twist(embodiment: EmbodimentCfg) -> None:
  scene = Scene(SceneCfg(), embodiment.build())
  data = mujoco.MjData(scene.model)
  assert np.array_equal(
    scene.twists(data.qpos[None].copy(), data.qvel[None].copy()),
    np.zeros((1, len(scene.hands), 6)),
  )
