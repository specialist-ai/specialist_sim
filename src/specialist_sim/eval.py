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

__all__ = ["LIFT", "Agent", "Outcome", "summary", "sweep"]

LIFT = 0.010
"""How far off the table the cube must come for an episode to count as a pick."""


class Agent(Protocol):
  """What the benchmark asks of a policy.

  ``reset`` names exactly the worlds being restarted: when the bench restarts
  one world mid-batch, the agent must clear that world's state and no other.
  ``act`` answers for every world at once, for the hands the task drives.
  """

  def reset(self, worlds: Sequence[int]) -> None: ...

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

  @property
  def picked(self) -> bool:
    """Whether the cube was carried clear of the table, rather than nudged."""
    return self.lift > LIFT and self.held_steps > 0

  @property
  def clean(self) -> bool:
    return self.success and self.picked

  @property
  def bucket(self) -> str:
    """One of ``clean``, ``slid``, ``lost``, ``neither``: solved by a carry,
    solved by a nudge, carried but not solved, or nothing at all."""
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
  if env.cfg.max_steps <= budget:
    raise ValueError(
      f"the environment truncates at {env.cfg.max_steps} steps, inside the "
      f"budget of {budget}: the bench must own the ending"
    )
  queue = list(int(seed) for seed in seeds)
  first = queue[: env.worlds] or [0]
  padded = first + [first[-1]] * (env.worlds - len(first))
  del queue[: len(first)]
  observation, info = env.reset(seeds=padded)
  agent.reset(range(env.worlds))
  rest = env.scene.cube.rest
  running: list[int | None] = list(first) + [None] * (env.worlds - len(first))
  steps = np.zeros(env.worlds, np.int64)
  lift = np.zeros(env.worlds)
  held = np.zeros(env.worlds, np.int64)
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
        )
      )
      if queue:
        drawn = queue.pop(0)
        observation, info = env.reset(seeds=[drawn], worlds=[world])
        agent.reset([world])
        running[world] = drawn
        steps[world], lift[world], held[world] = 0, 0.0, 0
      else:
        running[world] = None
  return sorted(outcomes, key=lambda outcome: outcome.seed)


def summary(
  outcomes: Sequence[Outcome], env: PickPlaceEnv, budget: int
) -> dict[str, Any]:
  """The rates a benchmark reports, from the outcomes it produced, beside the
  protocol they were produced under: the budget, the camera lag and how many
  worlds ran at once, so two reports are known to be comparable."""
  wins = [outcome for outcome in outcomes if outcome.success]
  buckets = {name: 0 for name in ("clean", "slid", "lost", "neither")}
  for outcome in outcomes:
    buckets[outcome.bucket] += 1
  return {
    "episodes": len(outcomes),
    "budget": budget,
    "camera_lag": env.cfg.camera_lag,
    "worlds": env.worlds,
    "success_rate": len(wins) / len(outcomes),
    "clean_rate": buckets["clean"] / len(outcomes),
    "dropped_rate": sum(outcome.dropped for outcome in outcomes) / len(outcomes),
    "steps_to_success": (
      float(np.mean([outcome.steps for outcome in wins])) if wins else float("nan")
    ),
    "lift": float(np.median([outcome.lift for outcome in outcomes])),
    "buckets": buckets,
  }
