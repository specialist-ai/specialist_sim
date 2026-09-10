"""A shared benchmark: one episode protocol, one outcome, any policy.

The environment and a policy meet at ``Agent``, which any repo can satisfy
with a thin adapter; everything on this side of that seam — where episodes
start, when they end, and what an outcome is called — is defined once here,
so two repos reporting a rate are reporting the same quantity.

Episodes are scored from their first step: an agent that spends steps warming
up spends them out of its own budget.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from specialist_sim.action import hold
from specialist_sim.arrays import Float64Array
from specialist_sim.env import PickPlaceEnv
from specialist_sim.observation import flat_row
from specialist_sim.spaces import GRIPPERS, HORIZON, PER_HAND

__all__ = [
  "BUDGET",
  "LIFT",
  "LIFTS",
  "PROTOCOL",
  "Agent",
  "Outcome",
  "against",
  "summary",
  "sweep",
]

BUDGET = 100
"""How many control steps an episode is given, at ten a second."""

LIFT = 0.010
"""How far off the table the cube must come for an episode to count as a pick."""

LIFTS = (0.006, 0.008, 0.010, 0.012)
"""The lifts a summary also reports clean at, since the score turns on one."""

PROTOCOL = 5
"""The benchmark's revision, bumped by any change that moves an outcome for a
fixed seed, so a logged score says which protocol it came from.

Under this one every start is drawn from the episode's own stream by the task,
the cube turned off the tape as on it; a hand the task does not drive is held
at its commanded setpoint; the bench names each episode's seed to the agent; a
trajectory is tracked wherever it points; a geom is shaded the color the
visualizer resolves for it, whether that comes from a material or from an rgba
of its own; and the gripper is the described one, held against gravity by
whatever carries it rather than by a model that cancels its weight.
"""


class Agent(Protocol):
  """What the benchmark asks of a policy.

  ``reset`` names exactly the worlds being restarted and the seed each one
  runs: when the bench restarts one world mid-batch, the agent clears that
  world's state and no other, and its noise is a function of the seed alone.
  ``act`` answers for every world at once, for the hands the task drives.
  """

  def reset(self, worlds: Sequence[int], seeds: Sequence[int]) -> None: ...

  def act(self, observation: dict[str, Any]) -> Float64Array: ...


@dataclass(frozen=True)
class Outcome:
  """What one episode did, computed from the environment's own reports."""

  seed: int
  success: bool
  dropped: bool
  steps: int
  lift: float
  held_steps: int
  broke: bool
  tracking: float
  """The median over steps of the largest gap between a driven hand and its
  setpoint at the step's end, in meters."""

  @property
  def picked(self) -> bool:
    """Whether the cube was carried clear of the table, rather than nudged."""
    return self.picked_at(LIFT)

  def picked_at(self, lift: float) -> bool:
    return self.lift > lift and self.held_steps > 0

  @property
  def clean(self) -> bool:
    return self.success and self.picked

  @property
  def bucket(self) -> str:
    """One of ``broke``, ``clean``, ``slid``, ``lost``, ``neither``: a world
    the physics could not carry, solved by a carry, solved by a nudge, carried
    but not solved, or nothing at all."""
    if self.broke:
      return "broke"
    if self.success:
      return "clean" if self.picked else "slid"
    return "lost" if self.picked else "neither"


def sweep(
  env: PickPlaceEnv, agent: Agent, seeds: Sequence[int], budget: int
) -> list[Outcome]:
  """Run one episode per seed, keeping every world busy, and score them all.

  Where an episode opens and what ends it are the environment's task. A world
  whose episode ends takes the next seed from the pool and restarts alone; the
  agent is told exactly which world was restarted.
  """
  if env.cfg.max_steps is not None and env.cfg.max_steps <= budget:
    raise ValueError(
      f"the environment truncates at {env.cfg.max_steps} steps, inside the "
      f"budget of {budget}: the bench must own the ending"
    )
  queue = list(int(seed) for seed in seeds)
  first = queue[: env.worlds] or [0]
  padded = first + [first[-1]] * (env.worlds - len(first))
  del queue[: len(first)]
  observation, info = env.reset(seeds=padded)
  agent.reset(range(env.worlds), padded)
  rest = env.scene.cube.rest
  running: list[int | None] = list(first) + [None] * (env.worlds - len(first))
  steps = np.zeros(env.worlds, np.int64)
  lift = np.zeros(env.worlds)
  held = np.zeros(env.worlds, np.int64)
  broke = np.zeros(env.worlds, np.bool_)
  tracking: list[list[float]] = [[] for _ in range(env.worlds)]
  outcomes: list[Outcome] = []
  while any(seed is not None for seed in running):
    action = np.asarray(agent.act(observation), np.float64).copy()
    for world, seed in enumerate(running):
      if seed is None:
        action[world] = hold(flat_row(observation, world=world))
    observation, _, terminated, _, info = env.step(action)
    for world, seed in enumerate(running):
      if seed is None:
        continue
      steps[world] += 1
      lift[world] = max(lift[world], float(info["cube_pos"][world, 2]) - rest)
      held[world] += bool(info["held"][world])
      broke[world] |= bool(info["broke"][world])
      tracking[world].append(float(info["tracking"][world]))
      if not (bool(terminated[world]) or steps[world] >= budget):
        continue
      outcomes.append(
        Outcome(
          seed=seed,
          success=bool(info["is_success"][world]),
          dropped=bool(info["dropped"][world]),
          steps=int(steps[world]),
          lift=float(lift[world]),
          held_steps=int(held[world]),
          broke=bool(broke[world]),
          tracking=float(np.median(tracking[world])),
        )
      )
      if queue:
        drawn = queue.pop(0)
        observation, info = env.reset(seeds=[drawn], worlds=[world])
        agent.reset([world], [drawn])
        running[world] = drawn
        steps[world], lift[world], held[world], broke[world] = 0, 0.0, 0, False
        tracking[world] = []
      else:
        running[world] = None
  return sorted(outcomes, key=lambda outcome: outcome.seed)


def summary(
  outcomes: Sequence[Outcome], env: PickPlaceEnv, budget: int
) -> dict[str, Any]:
  """The rates a benchmark reports, from the outcomes it produced, beside
  everything they were produced under — the protocol, the environment's
  revision, the task, the budget, the camera lag, the worlds — so two reports
  are known to be comparable before their rates are."""
  task = env.cfg.task
  wins = [outcome for outcome in outcomes if outcome.success]
  buckets = {name: 0 for name in ("broke", "clean", "slid", "lost", "neither")}
  for outcome in outcomes:
    buckets[outcome.bucket] += 1
  return {
    "protocol": PROTOCOL,
    "revision": env.revision,
    "driven": task.driven,
    "lenses": task.lenses,
    "start": repr(task.start),
    "goal": repr(task.goal),
    "episodes": len(outcomes),
    "budget": budget,
    "camera_lag": env.cfg.camera_lag,
    "worlds": env.worlds,
    "success_rate": len(wins) / len(outcomes),
    "clean_rate": buckets["clean"] / len(outcomes),
    "dropped_rate": sum(outcome.dropped for outcome in outcomes) / len(outcomes),
    "broke_rate": sum(outcome.broke for outcome in outcomes) / len(outcomes),
    "steps_to_success": (
      float(np.mean([outcome.steps for outcome in wins])) if wins else float("nan")
    ),
    "lift": float(np.median([outcome.lift for outcome in outcomes])),
    "tracking": float(np.median([outcome.tracking for outcome in outcomes])),
    "clean_at_lift": {
      lift: sum(o.success and o.picked_at(lift) for o in outcomes) / len(outcomes)
      for lift in LIFTS
    },
    "buckets": buckets,
  }


def against(
  env: PickPlaceEnv,
  agent: Agent,
  driven: Sequence[str],
  seeds: Sequence[int],
  budget: int,
) -> list[float]:
  """How far the agent's first setpoint sits from the expert's, in the states
  the expert drives to.

  The expert drives, so every state the agent answers from is one the corpus
  is full of; the gap is what the agent gets wrong where it was taught, apart
  from anywhere its own choices would have carried it. One gap per driven
  hand per plan, in meters.
  """
  from specialist_sim.expert import Expert

  hands = [GRIPPERS.index(slot) for slot in driven]
  gaps: list[float] = []
  for first in range(0, len(seeds), env.worlds):
    chunk = [int(seed) for seed in seeds[first : first + env.worlds]]
    padded = chunk + [chunk[-1]] * (env.worlds - len(chunk))
    observation, info = env.reset(seeds=padded)
    agent.reset(range(env.worlds), padded)
    experts = [Expert(env, np.random.default_rng(seed)) for seed in padded]
    live = np.arange(len(chunk))
    for _ in range(budget):
      planned = np.asarray(agent.act(observation), np.float64).reshape(
        env.worlds, HORIZON, len(hands), PER_HAND
      )
      action = np.stack(
        [
          expert.act(flat_row(observation, world=world), info["cube_pos"][world])
          for world, expert in enumerate(experts)
        ]
      ).reshape(env.worlds, HORIZON, len(GRIPPERS), PER_HAND)
      for world in live:
        gaps += [
          float(np.linalg.norm(planned[world, 0, i, :3] - action[world, 0, hand, :3]))
          for i, hand in enumerate(hands)
        ]
      observation, _, terminated, _, info = env.step(
        action.reshape(env.worlds, HORIZON, -1)
      )
      live = live[~terminated[live]]
      if not len(live):
        break
  return gaps
