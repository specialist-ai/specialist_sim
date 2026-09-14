"""The benchmark: one episode protocol, one outcome, any policy, any scene.
An episode is scored from its first step, and the bench owns the budget.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import specialist_description

from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.gripper import VARIANT
from specialist_sim.rig import Observation
from specialist_sim.trajectory import hold

__all__ = [
  "BUDGET",
  "Agent",
  "Episodes",
  "Outcome",
  "Report",
  "revision",
  "summary",
  "sweep",
]

BUDGET = 100
"""How many control steps an episode is given."""


class Report(Protocol):
  """The least a scene says of every world after a reset or a step."""

  @property
  def success(self) -> BoolArray: ...
  @property
  def done(self) -> BoolArray: ...
  @property
  def broke(self) -> BoolArray: ...
  @property
  def tracking(self) -> Float64Array: ...
  @property
  def labels(self) -> Mapping[str, BoolArray]:
    """Whatever else the scene counts about each world, by its own name."""
    ...


class Episodes(Protocol):
  """A scene environment as the bench drives it."""

  @property
  def worlds(self) -> int: ...
  @property
  def cfg(self) -> object: ...
  @property
  def assets(self) -> Sequence[Path]:
    """The directories whose files decide what the scene is."""
    ...

  def reset(
    self, seeds: Sequence[int], worlds: Sequence[int] | None = None
  ) -> tuple[Observation, Report]: ...

  def step(self, action: Float64Array) -> tuple[Observation, Report]: ...


class Agent(Protocol):
  def reset(self, worlds: Sequence[int], seeds: Sequence[int]) -> None:
    """Clear the named worlds' state; each runs the seed beside it."""
    ...

  def act(self, observation: Observation) -> Float64Array:
    """A trajectory for every world, ``(worlds, poses, driven, 8)``."""
    ...


@dataclass(frozen=True)
class Outcome:
  """What one episode did, from the scene's own reports."""

  seed: int
  success: bool
  steps: int
  broke: bool
  tracking: float
  """The median over steps of the largest gap between a driven hand and its
  setpoint, in metres."""
  labels: Mapping[str, bool]
  """Every name the scene counts, and whether it held when the episode ended,
  so a report names the same rates whatever happened to run."""


def sweep(
  env: Episodes, agent: Agent, seeds: Sequence[int], budget: int = BUDGET
) -> list[Outcome]:
  """Run one episode per seed, keeping every world busy: a world whose
  episode ends restarts alone on the next seed, and the agent is told."""
  queue = [int(seed) for seed in seeds]
  if not queue:
    return []
  first = queue[: env.worlds]
  padded = first + [first[-1]] * (env.worlds - len(first))
  del queue[: len(first)]
  observation, _ = env.reset(padded)
  agent.reset(range(env.worlds), padded)
  running: list[int | None] = list(first) + [None] * (env.worlds - len(first))
  steps = np.zeros(env.worlds, np.int64)
  tracking: list[list[float]] = [[] for _ in range(env.worlds)]
  outcomes: list[Outcome] = []
  while any(seed is not None for seed in running):
    action = np.asarray(agent.act(observation), np.float64).copy()
    for world, seed in enumerate(running):
      if seed is None:
        action[world] = hold(observation.state[world, -1], action.shape[1])
    observation, report = env.step(action)
    for world, seed in enumerate(running):
      if seed is None:
        continue
      steps[world] += 1
      tracking[world].append(float(report.tracking[world]))
      if not (bool(report.done[world]) or steps[world] >= budget):
        continue
      outcomes.append(
        Outcome(
          seed=seed,
          success=bool(report.success[world]),
          steps=int(steps[world]),
          broke=bool(report.broke[world]),
          tracking=float(np.median(tracking[world])),
          labels={name: bool(held[world]) for name, held in report.labels.items()},
        )
      )
      if queue:
        drawn = queue.pop(0)
        observation, _ = env.reset([drawn], worlds=[world])
        agent.reset([world], [drawn])
        running[world] = drawn
        steps[world], tracking[world] = 0, []
      else:
        running[world] = None
  return sorted(outcomes, key=lambda outcome: outcome.seed)


def summary(outcomes: Sequence[Outcome], env: Episodes, budget: int) -> dict[str, Any]:
  """The rates a benchmark reports, beside everything they were produced
  under, so two reports are known to be comparable before their rates are."""
  wins = [outcome for outcome in outcomes if outcome.success]
  named = sorted({name for outcome in outcomes for name in outcome.labels})
  return {
    "revision": revision(env),
    "cfg": repr(env.cfg),
    "episodes": len(outcomes),
    "budget": budget,
    "worlds": env.worlds,
    "success_rate": len(wins) / len(outcomes),
    "broke_rate": sum(outcome.broke for outcome in outcomes) / len(outcomes),
    **{
      f"{name}_rate": sum(outcome.labels[name] for outcome in outcomes) / len(outcomes)
      for name in named
    },
    "steps_to_success": (
      float(np.mean([outcome.steps for outcome in wins])) if wins else float("nan")
    ),
    "tracking": float(np.median([outcome.tracking for outcome in outcomes])),
  }


def revision(env: Episodes) -> str:
  """A hash of what decides an environment's answer to a seed: the gripper's
  description, the configuration, MuJoCo, and the scene's assets."""
  digest = hashlib.sha256()
  for root in (specialist_description.variant_dir(VARIANT), *env.assets):
    for path in sorted(Path(root).rglob("*")):
      if path.is_file():
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
  digest.update(repr(env.cfg).encode())
  digest.update(mujoco.__version__.encode())
  return digest.hexdigest()[:12]
