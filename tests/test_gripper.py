"""The hand: what the description gives it, and how a carrier stands it up."""

import mujoco
import numpy as np
import pytest

from specialist_sim import gripper
from specialist_sim.embodiment import Embodiment, Floating
from specialist_sim.gripper import SLOTS
from world import bare


def _stand(embodiment: Embodiment) -> mujoco.MjModel:
  spec = bare()
  embodiment.attach(spec)
  return spec.compile()


def test_the_described_hand_binds_by_its_slot() -> None:
  model = _stand(Floating())
  for slot in SLOTS:
    hand = gripper.bind(model, slot)
    assert hand.slot == slot
    assert model.body(hand.base).name == f"{slot}_base"
    assert len(gripper.pads(model, hand)) > 0, "a grasp needs pads to touch with"
  offset = gripper.grasp(model, gripper.bind(model, SLOTS[0]))
  assert 0.05 < np.linalg.norm(offset) < 0.15
  assert gripper.SHUT < gripper.OPEN


def test_a_carrier_stands_the_hands_it_was_given(embodiment: Embodiment) -> None:
  model = _stand(embodiment)
  hands = embodiment.bind(model)
  assert tuple(hand.slot for hand in hands) == SLOTS
  one = Floating(slots=(SLOTS[1],))
  model = _stand(one)
  assert tuple(hand.slot for hand in one.bind(model)) == (SLOTS[1],)


def test_a_hand_is_read_back_where_its_carrier_put_it(embodiment: Embodiment) -> None:
  """A pose is the gripper base's, whatever holds it up."""
  model = _stand(embodiment)
  hands = embodiment.bind(model)
  data, scratch = mujoco.MjData(model), mujoco.MjData(model)
  states = np.array(
    [
      [0.47, 0.16, -0.08, 1.0, 0.0, 0.0, 0.0, 0.3],
      [0.47, -0.16, -0.08, 0.0, 0.0, 0.0, 1.0, 0.1],
    ]
  )
  for index, (hand, state) in enumerate(zip(hands, states, strict=True)):
    embodiment.place(data, index, state[:7])
    data.qpos[list(hand.fingers)] = state[7], -state[7]
  read = gripper.states(model, scratch, hands, data.qpos[None])[0]
  assert read[:, :3] == pytest.approx(states[:, :3], abs=1e-6)
  assert np.abs(read[:, 3:7]) == pytest.approx(np.abs(states[:, 3:7]), abs=1e-6)
  assert read[:, 7] == pytest.approx(states[:, 7])
  twists = gripper.twists(
    model, scratch, hands, data.qpos[None], np.zeros((1, model.nv))
  )
  assert twists == pytest.approx(0.0, abs=1e-9)


def test_a_free_hand_reports_the_twist_its_joint_carries_it_with() -> None:
  floating = Floating()
  model = _stand(floating)
  hands = floating.bind(model)
  data, scratch = mujoco.MjData(model), mujoco.MjData(model)
  data.qvel[:] = np.random.default_rng(0).normal(size=model.nv) * 0.1
  mujoco.mj_kinematics(model, data)
  twists = gripper.twists(model, scratch, hands, data.qpos[None], data.qvel[None])
  for index, slot in enumerate(SLOTS):
    rig = model.body(f"{slot}_rig")
    dof = int(rig.dofadr[0])
    spin = data.xmat[rig.id].reshape(3, 3) @ data.qvel[dof + 3 : dof + 6]
    assert twists[0, index] == pytest.approx(
      [*data.qvel[dof : dof + 3], *spin], abs=1e-12
    )
