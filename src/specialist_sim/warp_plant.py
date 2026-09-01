"""Every world's physics, advanced a control step at a time on the GPU.

Answers ``plant.Plant``'s contract through ``mujoco_warp``, which the ``gpu``
extra installs. A control step is captured once as a CUDA graph and replayed, so
the impedance rides a kernel inside it rather than a host call between substeps.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

import mujoco
import numpy as np
import warp as wp

from specialist_sim.action import Hands
from specialist_sim.plant import STATE
from specialist_sim.scene import Gripper

__all__ = ["WarpPlant"]

CONTACT_HEADROOM = 192
"""Contacts per world the batched buffers hold; the plant peaks near 113."""
CONSTRAINT_HEADROOM = 128
"""Constraint rows per world; the plant peaks near 86."""
CCD_HEADROOM = 96
"""Convex-convex contacts per world; sized down from mujoco_warp's default,
whose EPA scratch would otherwise grow with the square of the world count."""


class WarpPlant:
  """Every world's physics, stepped together on the GPU through mujoco_warp.

  A control step is captured once as a CUDA graph and replayed, so the impedance
  rides a kernel inside it rather than a host call between substeps. Nothing can
  read the worlds part way through a replay, so the lenses are read at the end:
  the capture row must be the last one, which is what a camera lag of a whole
  number of rows asks for.
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
    import mujoco_warp as mjw

    if capture != rows:
      raise ValueError(
        f"the GPU reads the lenses at the end of a step, so the capture row is "
        f"{rows}, not {capture}: a camera lag must be a whole number of rows"
      )
    self.model, self.datas = model, list(datas)
    self.rows, self._per_row = rows, per_row
    self._grippers, self._hands = list(grippers), hands
    worlds, count = len(self.datas), len(self._grippers)

    # mujoco_warp cannot factor non-uniform damping within a free joint; carry
    # the rotational value uniformly and reapply the linear surplus in the wrench.
    device_model = copy.deepcopy(model)
    surpluses = set()
    for grip in self._grippers:
      adr = device_model.body_dofadr[grip.rig]
      linear, angular = device_model.dof_damping[adr], device_model.dof_damping[adr + 3]
      device_model.dof_damping[adr : adr + 6] = angular
      surpluses.add(float(linear - angular))
    if len(surpluses) != 1:
      raise ValueError(f"the grippers disagree on damping: {sorted(surpluses)}")
    self._surplus = surpluses.pop()
    device_model.opt.ls_iterations = max(device_model.opt.ls_iterations, 50)

    seed = self.datas[0]
    mujoco.mj_forward(model, seed)
    self._device_model = mjw.put_model(device_model)
    # nconmax and njmax are per world: contacts share one array of
    # nconmax * nworld and the constraint arrays are shaped (nworld, njmax), so
    # scaling them here as well would make the arena quadratic in the batch.
    self.data = mjw.put_data(
      device_model,
      seed,
      nworld=worlds,
      nconmax=max(CONTACT_HEADROOM, int(seed.ncon) + 8),
      njmax=max(CONSTRAINT_HEADROOM, int(seed.nefc) + 32),
      naccdmax=CCD_HEADROOM * worlds,
    )

    substeps = hands.substeps
    self._ramp_position = wp.zeros((worlds, substeps, count, 3), dtype=wp.float32)
    self._ramp_quaternion = wp.zeros((worlds, substeps, count, 4), dtype=wp.float32)
    self._ramp_opening = wp.zeros((worlds, substeps, count), dtype=wp.float32)
    self._ramp_twist = wp.zeros((worlds, substeps, count, 6), dtype=wp.float32)
    self._rig = wp.array([grip.rig for grip in self._grippers], dtype=wp.int32)
    self._servo = wp.array(
      [list(grip.servos) for grip in self._grippers], dtype=wp.int32
    )
    self._linear_dof = wp.array(
      [int(model.body_dofadr[grip.rig]) for grip in self._grippers], dtype=wp.int32
    )
    self._qpos_adr = wp.array([grip.qpos for grip in self._grippers], dtype=wp.int32)
    self._finger_adr = wp.array(
      [grip.fingers[0] for grip in self._grippers], dtype=wp.int32
    )
    self._sampled = wp.zeros((worlds, count, rows, 8), dtype=wp.float32)
    self._graph = None

    self._state = np.zeros((worlds, mujoco.mj_stateSize(model, STATE)))
    self.qpos = self._state[:, 1 : 1 + model.nq]
    self.qvel = self._state[:, 1 + model.nq : 1 + model.nq + model.nv]
    self.poses = np.zeros((worlds, count, rows, 7), np.float32)
    self.openings = np.zeros((worlds, count, rows), np.float32)

  def gather(self) -> None:
    for world, data in enumerate(self.datas):
      mujoco.mj_getState(self.model, data, self._state[world], STATE)
    self.data.qpos.assign(np.ascontiguousarray(self.qpos, np.float32))
    self.data.qvel.assign(np.ascontiguousarray(self.qvel, np.float32))

  def land(self) -> None:
    self.qpos[:] = self.data.qpos.numpy()
    self.qvel[:] = self.data.qvel.numpy()
    for world, data in enumerate(self.datas):
      mujoco.mj_setState(self.model, data, self._state[world], STATE)
      mujoco.mj_forward(self.model, data)

  def run(self, on_capture: Callable[[], None]) -> None:
    self.gather()
    position, quaternion, opening, twist = self._hands.ramps
    self._ramp_position.assign(np.ascontiguousarray(position, np.float32))
    self._ramp_quaternion.assign(np.ascontiguousarray(quaternion, np.float32))
    self._ramp_opening.assign(np.ascontiguousarray(opening, np.float32))
    self._ramp_twist.assign(np.ascontiguousarray(twist, np.float32))
    if self._graph is None:
      with wp.ScopedCapture() as capture:
        self._replay()
      self._graph = capture.graph
    wp.capture_launch(self._graph)
    sampled = self._sampled.numpy()
    self.poses[:] = sampled[..., :7]
    self.openings[:] = sampled[..., 7]
    self.land()
    on_capture()

  def _replay(self) -> None:
    """The substeps of one control step, unrolled so each names its own row."""
    import mujoco_warp as mjw

    cfg = self._hands.cfg
    shape = (len(self.datas), len(self._grippers))
    for substep in range(self._hands.substeps):
      wp.launch(
        _pull,
        dim=shape,
        inputs=[
          substep,
          self._rig,
          self._servo,
          self._linear_dof,
          self._ramp_position,
          self._ramp_quaternion,
          self._ramp_opening,
          self._ramp_twist,
          cfg.stiffness,
          cfg.damping,
          cfg.rotational_stiffness,
          cfg.rotational_damping,
          self._surplus,
          self.data.xpos,
          self.data.xmat,
          self.data.qvel,
          self.data.ctrl,
          self.data.xfrc_applied,
        ],
      )
      mjw.step(self._device_model, self.data)
      if (substep + 1) % self._per_row:
        continue
      wp.launch(
        _read_row,
        dim=shape,
        inputs=[
          (substep + 1) // self._per_row - 1,
          self._qpos_adr,
          self._finger_adr,
          self.data.qpos,
          self._sampled,
        ],
      )


@wp.kernel
def _pull(
  substep: int,
  rig: wp.array(dtype=wp.int32),
  servo: wp.array2d(dtype=wp.int32),
  linear_dof: wp.array(dtype=wp.int32),
  ramp_position: wp.array4d(dtype=wp.float32),
  ramp_quaternion: wp.array4d(dtype=wp.float32),
  ramp_opening: wp.array3d(dtype=wp.float32),
  ramp_twist: wp.array4d(dtype=wp.float32),
  stiffness: wp.float32,
  damping: wp.float32,
  rotational_stiffness: wp.float32,
  rotational_damping: wp.float32,
  surplus: wp.float32,
  xpos: wp.array2d(dtype=wp.vec3),
  xmat: wp.array2d(dtype=wp.mat33),
  qvel: wp.array2d(dtype=wp.float32),
  ctrl: wp.array2d(dtype=wp.float32),
  xfrc: wp.array2d(dtype=wp.spatial_vector),
):
  """The wrench and servo angles one substep of the ramp asks of every world."""
  world, hand = wp.tid()
  body = rig[hand]

  angle = ramp_opening[world, substep, hand]
  ctrl[world, servo[hand, 0]] = angle
  ctrl[world, servo[hand, 1]] = -angle

  # A ramp holds wxyz, as mujoco does; a warp quaternion is xyzw.
  want = wp.quat(
    ramp_quaternion[world, substep, hand, 1],
    ramp_quaternion[world, substep, hand, 2],
    ramp_quaternion[world, substep, hand, 3],
    ramp_quaternion[world, substep, hand, 0],
  )
  error = want * wp.quat_inverse(wp.quat_from_matrix(xmat[world, body]))
  sine = wp.sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2])
  # The angle comes from atan2 rather than the matrix's antisymmetric part, so a
  # half turn is as sound as none and a far pose cannot ask for a boundless torque.
  swept = 2.0 * wp.atan2(sine, error[3])
  if swept > wp.pi:
    swept = swept - 2.0 * wp.pi
  scale = wp.float32(0.0)
  if sine > 1.0e-14:
    scale = swept / sine
  turn = wp.vec3(error[0] * scale, error[1] * scale, error[2] * scale)

  gap = (
    wp.vec3(
      ramp_position[world, substep, hand, 0],
      ramp_position[world, substep, hand, 1],
      ramp_position[world, substep, hand, 2],
    )
    - xpos[world, body]
  )
  feed_linear = wp.vec3(
    ramp_twist[world, substep, hand, 0],
    ramp_twist[world, substep, hand, 1],
    ramp_twist[world, substep, hand, 2],
  )
  feed_angular = wp.quat_rotate(
    want,
    wp.vec3(
      ramp_twist[world, substep, hand, 3],
      ramp_twist[world, substep, hand, 4],
      ramp_twist[world, substep, hand, 5],
    ),
  )
  adr = linear_dof[hand]
  drag = (
    wp.vec3(qvel[world, adr + 0], qvel[world, adr + 1], qvel[world, adr + 2]) * surplus
  )
  force = gap * stiffness + feed_linear * damping - drag
  torque = turn * rotational_stiffness + feed_angular * rotational_damping
  xfrc[world, body] = wp.spatial_vector(
    force[0], force[1], force[2], torque[0], torque[1], torque[2]
  )


@wp.kernel
def _read_row(
  row: int,
  qpos_adr: wp.array(dtype=wp.int32),
  finger_adr: wp.array(dtype=wp.int32),
  qpos: wp.array2d(dtype=wp.float32),
  sampled: wp.array4d(dtype=wp.float32),
):
  """Keep every world's hands where one row of the step timeline finds them."""
  world, hand = wp.tid()
  adr = qpos_adr[hand]
  for offset in range(7):
    sampled[world, hand, row, offset] = qpos[world, adr + offset]
  sampled[world, hand, row, 7] = qpos[world, finger_adr[hand]]
