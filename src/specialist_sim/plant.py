"""Every world's physics, advanced a control step at a time on the CPU.

The worlds' ``MjData`` are the truth between control steps: a plant gathers that
state, advances every world one control step, samples the hands on the row
lattice, and lands the state back. ``warp_plant.WarpPlant`` answers the same
contract on the GPU, and which one an environment holds is all that separates
the two.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import mujoco
import numpy as np
from mujoco import rollout

from specialist_sim.action import Hands
from specialist_sim.scene import Gripper

__all__ = ["Plant"]

STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS
"""What carries from one substep to the next. A rollout names no other state, so
the solver starts every substep from a zero warmstart, and a run of single
substeps drifts from an unbroken one."""

CONTROL = mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_XFRC_APPLIED
"""What a substep is driven by: the finger servos, and the wrench on each rig."""


class Plant:
  """Every world's physics, stepped together through mujoco's thread pool.

  Stepping goes one substep at a time: the impedance answers to where each plant
  is, so the pool cannot be handed a whole control step.
  """

  def __init__(
    self,
    model: mujoco.MjModel,
    datas: Sequence[mujoco.MjData],
    grippers: Sequence[Gripper],
    rows: int,
    per_row: int,
    capture: int,
    hands: Hands,
  ):
    self.model, self.datas = model, list(datas)
    self.rows, self._per_row, self._capture = rows, per_row, capture
    self._grippers, self._hands = list(grippers), hands
    worlds, count = len(self.datas), len(self._grippers)
    threads = min(worlds, os.cpu_count() or 1)
    self._roll = rollout.Rollout(nthread=threads)
    self._scratch = [mujoco.MjData(model) for _ in range(threads)]
    self._state = np.zeros((worlds, mujoco.mj_stateSize(model, STATE)))
    self._stepped = np.zeros((worlds, 1, self._state.shape[1]))
    self._warmstart = np.zeros((worlds, model.nv))
    """Zeros, always: rollout otherwise reuses whatever its scratch data held,
    which would make a world's step depend on which thread ran it last."""
    self._control = np.zeros((worlds, 1, mujoco.mj_stateSize(model, CONTROL)))
    self.ctrl = self._control[:, 0, : model.nu]
    self.xfrc = self._control[:, 0, model.nu :].reshape(worlds, model.nbody, 6)
    """Views a command is written straight into, one row per world."""
    self.qpos = self._state[:, 1 : 1 + model.nq]
    self.qvel = self._state[:, 1 + model.nq : 1 + model.nq + model.nv]
    """Views of every world's joint coordinates and velocities, current while
    a control step runs."""
    self.poses = np.zeros((worlds, count, rows, 7), np.float32)
    self.openings = np.zeros((worlds, count, rows), np.float32)
    """Where every world's hands stood on each row of the step timeline."""

  def gather(self) -> None:
    """Take the state the worlds hold, which stepping runs from."""
    for world, data in enumerate(self.datas):
      mujoco.mj_getState(self.model, data, self._state[world], STATE)

  def land(self) -> None:
    """Put the stepped state back on the worlds, so a reader sees what ran."""
    for world, data in enumerate(self.datas):
      mujoco.mj_setState(self.model, data, self._state[world], STATE)
      mujoco.mj_forward(self.model, data)

  def run(self, on_capture: Callable[[], None]) -> None:
    """Advance every world one control step, reading the lenses at the capture row."""
    self.gather()
    for substep in range(self._hands.substeps):
      self._hands.command(self.qpos, substep, self.ctrl, self.xfrc)
      self._roll.rollout(
        self.model,
        self._scratch,
        self._state,
        self._control,
        control_spec=int(CONTROL),
        nstep=1,
        state=self._stepped,
        initial_warmstart=self._warmstart,
      )
      self._state[:] = self._stepped[:, 0]
      if (substep + 1) % self._per_row:
        continue
      row = (substep + 1) // self._per_row
      for hand, grip in enumerate(self._grippers):
        self.poses[:, hand, row - 1] = self.qpos[:, grip.qpos : grip.qpos + 7]
        self.openings[:, hand, row - 1] = self.qpos[:, grip.fingers[0]]
      if row == self._capture:
        self.land()
        on_capture()
    if self._capture != self.rows:
      self.land()
