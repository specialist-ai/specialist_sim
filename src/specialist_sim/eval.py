"""Policy evaluation: one episode protocol, one outcome, any policy, any env.
An episode is scored from its first step, and the evaluation owns the time limit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np
import specialist_description

from specialist_sim.action import hold
from specialist_sim.arrays import Float64Array
from specialist_sim.env import Env, Task
from specialist_sim.gripper import VARIANT
from specialist_sim.rig import Observation

__all__ = [
  "MAX_EPISODE_STEPS",
  "Outcome",
  "Policy",
  "eval_policy",
  "mcnemar",
  "revision",
  "summary",
]

MAX_EPISODE_STEPS = 100
"""How many control steps an episode is given."""


class Policy(Protocol):
  def reset(self, env_ids: Sequence[int], seeds: Sequence[int]) -> None:
    """Clear the named envs' state. Each runs the seed beside it."""
    ...

  def predict_action_chunk(self, observation: Observation) -> Float64Array:
    """An action chunk for every env, ``(num_envs, chunk_size, driven, 8)``."""
    ...


@dataclass(frozen=True)
class Outcome:
  """What one episode did, from the env's own reports."""

  seed: int
  success: bool
  progress: float
  """How far the task came, from 0 to 1, when the episode ended."""
  steps: int
  unstable: bool
  tracking_error: float
  """The median over steps of the largest gap between a driven hand and its
  setpoint, in meters."""
  labels: Mapping[str, bool]
  """Every name the task counts, and whether it held when the episode ended,
  so a report names the same rates whatever happened to run."""


def eval_policy[T: Task](
  env: Env[T],
  policy: Policy,
  seeds: Sequence[int],
  max_episode_steps: int = MAX_EPISODE_STEPS,
) -> list[Outcome]:
  """Run one episode per seed, keeping every env busy: an env whose episode
  ends restarts alone on the next seed, and the policy is told."""
  queue = [int(seed) for seed in seeds]
  if not queue:
    return []
  first = queue[: env.num_envs]
  padded = first + [first[-1]] * (env.num_envs - len(first))
  del queue[: len(first)]
  observation, _ = env.reset(padded)
  policy.reset(range(env.num_envs), padded)
  running: list[int | None] = list(first) + [None] * (env.num_envs - len(first))
  steps = np.zeros(env.num_envs, np.int64)
  tracking_error: list[list[float]] = [[] for _ in range(env.num_envs)]
  outcomes: list[Outcome] = []
  while any(seed is not None for seed in running):
    action = np.asarray(policy.predict_action_chunk(observation), np.float64).copy()
    for env_idx, seed in enumerate(running):
      if seed is None:
        action[env_idx] = hold(observation.state[env_idx, -1], action.shape[1])
    observation, report = env.step(action)
    for env_idx, seed in enumerate(running):
      if seed is None:
        continue
      steps[env_idx] += 1
      tracking_error[env_idx].append(float(report.tracking_error[env_idx]))
      if not (bool(report.done[env_idx]) or steps[env_idx] >= max_episode_steps):
        continue
      outcome = Outcome(
        seed=seed,
        success=bool(report.success[env_idx]),
        progress=float(report.progress[env_idx]),
        steps=int(steps[env_idx]),
        unstable=bool(report.unstable[env_idx]),
        tracking_error=float(np.median(tracking_error[env_idx])),
        labels={name: bool(held[env_idx]) for name, held in report.labels.items()},
      )
      outcomes.append(outcome)
      if queue:
        drawn = queue.pop(0)
        observation, _ = env.reset([drawn], env_ids=[env_idx])
        policy.reset([env_idx], [drawn])
        running[env_idx] = drawn
        steps[env_idx], tracking_error[env_idx] = 0, []
      else:
        running[env_idx] = None
  return sorted(outcomes, key=lambda outcome: outcome.seed)


def summary[T: Task](
  outcomes: Sequence[Outcome], env: Env[T], max_episode_steps: int
) -> dict[str, Any]:
  """The rates an evaluation reports, beside everything they were produced
  under, so two reports are known to be comparable before their rates are."""
  wins = [outcome for outcome in outcomes if outcome.success]
  named = sorted({name for outcome in outcomes for name in outcome.labels})
  return {
    "revision": revision(env),
    "task": repr(env.task),
    "cfg": repr(env.cfg),
    "n_episodes": len(outcomes),
    "max_episode_steps": max_episode_steps,
    "num_envs": env.num_envs,
    "success_rate": len(wins) / len(outcomes),
    "progress": float(np.mean([outcome.progress for outcome in outcomes])),
    "unstable_rate": sum(outcome.unstable for outcome in outcomes) / len(outcomes),
    **{
      f"{name}_rate": sum(outcome.labels[name] for outcome in outcomes) / len(outcomes)
      for name in named
    },
    "steps_to_success": (
      float(np.mean([outcome.steps for outcome in wins])) if wins else float("nan")
    ),
    "tracking_error": float(
      np.median([outcome.tracking_error for outcome in outcomes])
    ),
  }


def revision[T: Task](env: Env[T]) -> str:
  """A hash of what decides an environment's answer to a seed: the gripper's
  description, the task and what it counts, what it runs under, MuJoCo, and
  the task's assets."""
  digest = hashlib.sha256()
  for root in (specialist_description.get_variant_dir(VARIANT), *env.task.assets):
    for path in sorted(Path(root).rglob("*")):
      if path.is_file():
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
  digest.update(repr(env.task).encode())
  digest.update(f"{env.task.dwell} {env.task.latched}".encode())
  digest.update(repr(env.cfg).encode())
  digest.update(mujoco.__version__.encode())
  return digest.hexdigest()[:12]


def mcnemar(
  first: Sequence[Outcome], second: Sequence[Outcome]
) -> tuple[int, int, float]:
  """Two evaluations of the same seeds compared episode by episode: how many
  only the first succeeded at, how many only the second, and the exact
  two-sided McNemar p-value that they succeed equally often. Pairing removes
  the variance between seeds, so it detects a smaller difference than two
  success rates do."""
  wins = {outcome.seed: outcome.success for outcome in second}
  if sorted(wins) != sorted(outcome.seed for outcome in first):
    raise ValueError("the two evaluations ran different seeds")
  only_first = sum(outcome.success and not wins[outcome.seed] for outcome in first)
  only_second = sum(wins[outcome.seed] and not outcome.success for outcome in first)
  flipped = only_first + only_second
  tail = sum(comb(flipped, k) for k in range(min(only_first, only_second) + 1))
  return only_first, only_second, min(1.0, 2.0 * tail / 2**flipped)
