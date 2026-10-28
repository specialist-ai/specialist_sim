"""What a physics setting costs and what it buys, on the LEGO sorting scene:
speed against real time, and how the bricks and the grasp behave. Each
candidate is a step length and a ``physics.Physics``; every scenario runs on
fixed seeds.

  uv run python scripts/physics_bench.py                       # every candidate below
  uv run python scripts/physics_bench.py --only rlft default

* speed: the scripted policy clearing a 12-brick pile with both hands, the most
  contacts an episode makes. The real-time factor and the 99th percentile wall
  time of a 150 Hz row, against the 6.7 ms a row has in real time. Also the
  stability check: MuJoCo's warnings, and the fastest any brick flew (a runaway
  env throws bricks at many m/s).
* hold: a brick grasped, lifted and shaken by +-5 cm at 1.5 Hz on every axis
  (4.4 m/s^2 at the peaks, a quick hand). How deep the pads sink into it, how
  far it slips in the grasp, whether it falls.
* rest: piled and spread bricks left alone, 1.5 s to come down and then 2 s.
  How fast the fastest creeps over the last second, how deep they sink into one
  another and the tray, and the fastest any brick moves.
* spin: a brick turning at 25 rad/s in free flight, about what the takes'
  tossed bricks turn at. How much of its spin is left 50 ms later; air leaves
  nearly all.
* land: a brick let go tilted 6 cm over a bin, moving 0.5 m/s. The fastest it
  turns in the 150 ms after it touches down; the footage has ~25 rad/s.
* task: the scripted policy on single trays, right hand. How many bricks it
  sorts.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Any

import mujoco
import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.gripper import GRIPPING, SLOTS
from specialist_sim.physics import Physics
from specialist_sim.rig import Clock
from specialist_sim.sort_lego import Cfg, SortLegoEnv, initial, scene
from specialist_sim.sort_lego import script as sorting

ROW_HZ = 150
SOFT = (0.9, 0.95, 0.001, 0.5, 2.0)
"""MuJoCo's default impedance."""
DEFAULT = Physics()
RLFT = Physics(
  integrator="euler",
  impratio=10.0,
  pad_solref=(0.007, 1.0),
  pad_solimp=SOFT,
  contact_solref=(0.004, 1.0),
  contact_solimp=SOFT,
  object_damping=3e-4,
)
"""specialist_rlft's settings, the reference the defaults are measured against."""
CANDIDATES: dict[str, tuple[int, Physics]] = {
  "rlft": (5, RLFT),
  "default": (10, DEFAULT),
  "substeps5": (
    5,
    replace(DEFAULT, pad_solref=(0.004, 1.0), contact_solref=(0.004, 1.0)),
  ),
  "substeps20": (
    20,
    replace(DEFAULT, pad_solref=(0.001, 1.0), contact_solref=(0.001, 1.0)),
  ),
  "euler": (10, replace(DEFAULT, integrator="euler")),
  "impratio10": (10, replace(DEFAULT, impratio=10.0)),
  "noslip2": (10, replace(DEFAULT, noslip_iterations=2)),
  "soft_pads": (10, replace(DEFAULT, pad_solref=(0.007, 1.0), pad_solimp=SOFT)),
  "damping3e-4": (10, replace(DEFAULT, object_damping=3e-4)),
}
"""Each candidate's physics steps in a 150 Hz row, and its settings: specialist_rlft's,
the default, and the default with one setting changed. A time constant stays twice the
step, the least MuJoCo allows."""


def _env(per_row: int, physics: Physics, control: bool, **cfg: Any) -> SortLegoEnv:
  """The scene without lenses, stepped a control step at a time for a policy,
  or a row at a time for a commanded path."""
  clock = Clock(
    physics_dt=1.0 / (ROW_HZ * per_row),
    per_row=per_row,
    per_action=3 if control else 1,
    per_step=36 if control else 1,
  )
  return SortLegoEnv(Cfg(clock=clock, physics=physics, render=False, **cfg))


def _masks(env: SortLegoEnv) -> tuple[BoolArray, BoolArray, BoolArray]:
  model = env.rig.model
  bricks = np.isin(
    model.geom_bodyid, [model.body(f"brick_{i}").id for i in range(len(env.cfg.colors))]
  )
  pads = model.geom_condim == GRIPPING
  roots = model.body_rootid[model.geom_bodyid]
  fixtures = np.isin(roots, [model.body("fixtures").id])
  return bricks, pads, fixtures


def _deepest(data: mujoco.MjData, first: BoolArray, second: BoolArray) -> float:
  """The deepest any geom of ``first`` sinks into one of ``second``, meters."""
  deepest = 0.0
  for contact in data.contact[: data.ncon]:
    one, other = contact.geom
    if (first[one] and second[other]) or (first[other] and second[one]):
      deepest = max(deepest, -contact.dist)
  return deepest


def _brick_dofs(env: SortLegoEnv) -> np.ndarray[Any, Any]:
  model = env.rig.model
  return np.array(
    [model.joint(f"brick_{i}_joint").dofadr[0] for i in range(len(env.cfg.colors))]
  )


def speed(per_row: int, physics: Physics) -> dict[str, Any]:
  env = _env(per_row, physics, control=True, mode="pile", bricks=12)
  dofs = np.add.outer(_brick_dofs(env), np.arange(3))
  rows, warnings, fastest = [], 0, 0.0
  for seed in (0, 1):
    observation, _ = env.reset([seed])
    policy = sorting.scripted_policy(env, 0, np.random.default_rng(seed))
    data = env.rig.datas[0]
    for _ in range(round(12.0 / env.cfg.clock.step_dt)):
      action = policy.predict_action_chunk(observation.state[0, -1], env.bricks(0))
      started = time.perf_counter()
      observation, _ = env.step(action[None])
      rows.append((time.perf_counter() - started) / env.cfg.clock.per_step)
      fastest = max(fastest, float(np.abs(data.qvel[dofs]).max()))
    warnings += int(sum(warning.number for warning in data.warning))
  each = np.array(rows)
  return {
    "realtime": (len(each) / ROW_HZ) / each.sum(),
    "p99_ms": 1e3 * float(np.percentile(each, 99)),
    "warnings": warnings,
    "thrown_m_s": fastest,
  }


def _eased(plan: list[tuple[float, Float64Array]], t: float) -> Float64Array:
  """A pose along ``plan``, ``(arrive, pose)`` pairs, smoothstepped between them."""
  if t <= plan[0][0]:
    return plan[0][1]
  for (t0, a), (t1, b) in zip(plan, plan[1:], strict=False):
    if t <= t1:
      part = (t - t0) / (t1 - t0)
      eased = part * part * (3.0 - 2.0 * part)
      out = a + eased * (b - a)
      out[3:7] = rotation.slerp(a[3:7], b[3:7], eased)
      return out
  return plan[-1][1]


def _in_hand(state: Float64Array, brick: Float64Array) -> Float64Array:
  """A brick's pose in a hand's frame, ``(7,)``."""
  inverse = rotation.conjugate(state[3:7])
  return np.concatenate(
    [
      rotation.rotate(inverse, brick[:3] - state[:3]),
      rotation.multiply(inverse, brick[3:7]),
    ]
  )


def hold(per_row: int, physics: Physics, seed: int) -> dict[str, Any]:
  env = _env(per_row, physics, control=False, mode="single")
  bricks, pads, _ = _masks(env)
  observation, _ = env.reset([seed])
  layout = env.layouts[0]
  assert layout is not None
  which = int(np.flatnonzero(layout.active)[0])
  homes = observation.state[0, -1].copy()
  grasp = np.array([*sorting.grasp_pose(env.bricks(0)[which], SLOTS[1]), sorting.OPEN])
  above, shut = grasp.copy(), grasp.copy()
  above[2] += sorting.ABOVE
  shut[7] = sorting.SHUT
  lifted = shut.copy()
  lifted[2] += 0.10
  plan = [
    (0.0, homes[1]),
    (0.8, above),
    (1.3, grasp),
    (1.4, grasp),
    (1.7, shut),
    (2.0, shut),
    (2.6, lifted),
  ]
  shake_from, shake_for, settle_for = 2.8, 2.0, 0.5
  sinks: list[float] = []
  start: Float64Array | None = None
  data = env.rig.datas[0]
  for row in range(round((shake_from + shake_for + settle_for) * ROW_HZ)):
    t = row / ROW_HZ
    target = homes.copy()
    target[1] = _eased(plan, t)
    if shake_from <= t < shake_from + shake_for:
      phase = 2.0 * np.pi * 1.5 * (t - shake_from)
      ramp = min(1.0, (t - shake_from) / 0.3)
      target[1, :3] += 0.05 * np.sin(phase + np.array([0.0, 2.1, 4.2])) * ramp
    observation, _ = env.step(target[None, None])
    if t >= 2.0:
      sinks.append(_deepest(data, pads, bricks))
    if start is None and t >= shake_from:
      start = _in_hand(observation.state[0, -1, 1], env.bricks(0)[which])
  assert start is not None
  end = _in_hand(observation.state[0, -1, 1], env.bricks(0)[which])
  slip = float(np.linalg.norm(end[:3] - start[:3]))
  turned = rotation.log(rotation.multiply(rotation.conjugate(start[3:]), end[3:]))
  return {
    "sink_mm": 1e3 * float(np.median(sinks)),
    "slip_mm": 1e3 * slip,
    "slip_deg": float(np.degrees(np.linalg.norm(turned))),
    "dropped": slip > 0.015,
  }


def rest(per_row: int, physics: Physics) -> dict[str, Any]:
  env = _env(per_row, physics, control=False)
  bricks, _, fixtures = _masks(env)
  dofs = _brick_dofs(env)
  hands = initial.starts(SLOTS)
  data = env.rig.datas[0]
  creeps, sinks, fastest = [], [], 0.0
  for mode, seed in (("pile", 0), ("pile", 1), ("pile", 2), ("drop", 0)):
    layout = initial.draw(mode, seed, env.cfg.colors, bricks=8)
    env.place(0, layout.poses, hands)
    env.rig.settle([0], 1.5)
    observation = env.rig.begin([0])
    before = env.bricks(0)
    for row in range(2 * ROW_HZ):
      if row == ROW_HZ:
        before = env.bricks(0)
      observation, _ = env.step(observation.state[:, -1:])
      fastest = max(
        fastest, float(np.abs(data.qvel[np.add.outer(dofs, np.arange(3))]).max())
      )
      if row % 15 == 0:
        sinks.append(
          max(_deepest(data, bricks, bricks), _deepest(data, bricks, fixtures))
        )
    creeps.append(
      float(np.linalg.norm(env.bricks(0)[:, :3] - before[:, :3], axis=1).max())
    )
  return {
    "creep_mm_s": 1e3 * float(np.max(creeps)),
    "rest_sink_mm": 1e3 * float(np.max(sinks)),
    "fastest_m_s": fastest,
  }


def spin(per_row: int, physics: Physics) -> dict[str, Any]:
  env = _env(per_row, physics, control=False)
  bricks = initial.parked(env.cfg.colors)
  bricks[0, :3] = [0.3, 0.3, 0.5]  # in the air, clear of everything
  env.place(0, bricks, initial.starts(SLOTS))
  observation = env.rig.begin([0])
  data = env.rig.datas[0]
  at = env.rig.model.joint("brick_0_joint").dofadr[0]
  spinning = np.array([25.0, 5.0, 3.0])
  data.qvel[at + 3 : at + 6] = spinning
  for _ in range(round(0.05 * ROW_HZ)):
    observation, _ = env.step(observation.state[:, -1:])
  return {
    "spin_left": float(
      np.linalg.norm(data.qvel[at + 3 : at + 6]) / np.linalg.norm(spinning)
    )
  }


def land(per_row: int, physics: Physics) -> dict[str, Any]:
  env = _env(per_row, physics, control=False)
  model, data = env.rig.model, env.rig.datas[0]
  brick = model.body("brick_0").id
  at = model.joint("brick_0_joint").dofadr[0]
  peaks = []
  for tilt in (0.4, -0.7, 1.1):
    bricks = initial.parked(env.cfg.colors)
    bricks[1:, 2] -= 0.5  # out of the way
    middle = scene.bin_center("red")
    bricks[0, :3] = [
      middle[0] + 0.03,
      middle[1],
      scene.DESK_TOP + scene.BIN_WALL + 0.06,
    ]
    bricks[0, 3:] = rotation.exp(np.array([tilt, 0.3 * tilt, 0.0]))
    env.place(0, bricks, initial.starts(SLOTS))
    observation = env.rig.begin([0])
    data.qvel[at : at + 3] = [-0.5, 0.0, 0.0]
    touched: int | None = None
    peak = 0.0
    for row in range(round(0.5 * ROW_HZ)):
      observation, _ = env.step(observation.state[:, -1:])
      pairs = data.contact.geom[: data.ncon]
      if touched is None and (model.geom_bodyid[pairs] == brick).any():
        touched = row
      if touched is not None and row - touched <= 0.15 * ROW_HZ:
        peak = max(peak, float(np.linalg.norm(data.qvel[at + 3 : at + 6])))
    peaks.append(peak)
  return {"land_rad_s": float(np.median(peaks))}


def task(per_row: int, physics: Physics) -> dict[str, Any]:
  env = _env(per_row, physics, control=True, mode="single", driven=(SLOTS[1],))
  sorted_, started = 0, 0
  for seed in range(6):
    observation, report = env.reset([seed])
    policy = sorting.scripted_policy(env, 0, np.random.default_rng(seed))
    for _ in range(round(15.0 / env.cfg.clock.step_dt)):
      observation, report = env.step(
        policy.predict_action_chunk(observation.state[0, -1], env.bricks(0))[None]
      )
      if report.done[0] or policy.done:
        break
    sorted_ += int(report.sorted[0])
    started += int(report.started[0])
  return {"task": f"{sorted_}/{started}"}


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--only", nargs="+", choices=sorted(CANDIDATES), default=list(CANDIDATES)
  )
  args = parser.parse_args()
  mujoco.set_mju_user_warning(lambda message: None)
  columns = ["realtime", "p99_ms", "warnings", "thrown_m_s", "sink_mm", "slip_mm", "slip_deg",
             "dropped", "creep_mm_s", "rest_sink_mm", "fastest_m_s", "spin_left", "land_rad_s",
             "task"]  # fmt: skip
  print(f"{'candidate':14s} " + " ".join(f"{column:>12s}" for column in columns))
  for name in args.only:
    per_row, physics = CANDIDATES[name]
    row = speed(per_row, physics)
    holds = [hold(per_row, physics, seed) for seed in range(4)]
    row |= {
      "sink_mm": float(np.median([each["sink_mm"] for each in holds])),
      "slip_mm": float(np.median([each["slip_mm"] for each in holds])),
      "slip_deg": float(np.median([each["slip_deg"] for each in holds])),
      "dropped": f"{sum(each['dropped'] for each in holds)}/{len(holds)}",
    }
    row |= rest(per_row, physics) | spin(per_row, physics) | land(per_row, physics)
    row |= task(per_row, physics)
    cells = [
      f"{row[c]:12.2f}" if isinstance(row[c], float) else f"{row[c]:>12}"
      for c in columns
    ]
    print(f"{name:14s} " + " ".join(cells), flush=True)


if __name__ == "__main__":
  main()
