"""Policy evaluation: one episode protocol, one outcome, any policy, any env."""

import hashlib
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mujoco
import numpy as np

import specialist_sim
from specialist_sim.arrays import Float64Array
from specialist_sim.env import Env, Observation


class Policy(Protocol):
  def reset(self, env_ids: Sequence[int], seeds: Sequence[int]) -> None:
    """Clear the named envs' memory. Each runs the seed beside it."""

  def predict_action_chunk(self, observation: Observation) -> Float64Array:
    """An action chunk for every env, ``(num_envs, chunk_size, driven, 8)``."""


@dataclass(frozen=True)
class Outcome:
  """What one episode did, from the env's own reports."""

  seed: int
  success: bool
  progress: float
  """How far the task came, from 0 to 1, when the episode ended."""
  time: float
  """Seconds the episode ran."""
  unstable: bool
  tracking_error: float
  """The median over steps of the largest gap between a driven hand and its
  setpoint, in meters."""
  labels: Mapping[str, bool]
  """Every label the task reports, and whether it held when the episode ended."""


def eval_policy(
  env: Env[Any], policy: Policy, seeds: Sequence[int], time_limit: float
) -> list[Outcome]:
  """Run one episode per seed, each for ``time_limit`` seconds counted in whole
  control steps, at least one, unless it ends sooner. An env whose episode ends
  restarts alone on the next seed, and the policy is told; an env with no seed
  left steps on the policy's chunks, unscored."""
  step_dt = env.cfg.clock.step_dt
  limit = max(1, round(time_limit / step_dt))
  queue = [int(seed) for seed in seeds]
  first = queue[: env.num_envs]
  padded = first + [first[-1]] * (env.num_envs - len(first))
  del queue[: len(first)]
  observation, _ = env.reset(seed=padded)
  policy.reset(env_ids=range(env.num_envs), seeds=padded)
  running: list[int | None] = list(first) + [None] * (env.num_envs - len(first))
  steps = np.zeros(env.num_envs, np.int64)
  tracking_error: list[list[float]] = [[] for _ in range(env.num_envs)]
  outcomes: list[Outcome] = []
  while any(seed is not None for seed in running):
    action = np.asarray(policy.predict_action_chunk(observation), np.float64)
    observation, info = env.step(action)
    for env_idx, seed in enumerate(running):
      if seed is None:
        continue
      steps[env_idx] += 1
      tracking_error[env_idx].append(float(info["tracking_error"][env_idx]))
      if not (bool(info["done"][env_idx]) or steps[env_idx] >= limit):
        continue
      outcomes.append(
        Outcome(
          seed=seed,
          success=bool(info["success"][env_idx]),
          progress=float(info["progress"][env_idx]),
          time=float(steps[env_idx] * step_dt),
          unstable=bool(info["unstable"][env_idx]),
          tracking_error=float(np.median(tracking_error[env_idx])),
          labels={name: bool(label[env_idx]) for name, label in info["labels"].items()},
        )
      )
      if queue:
        drawn = queue.pop(0)
        observation, _ = env.reset(seed=[drawn], env_ids=[env_idx])
        policy.reset(env_ids=[env_idx], seeds=[drawn])
        running[env_idx] = drawn
        steps[env_idx] = 0
        tracking_error[env_idx] = []
      else:
        running[env_idx] = None
  return sorted(outcomes, key=lambda outcome: outcome.seed)


def summary(
  outcomes: Sequence[Outcome], env: Env[Any], time_limit: float
) -> dict[str, Any]:
  """The rates an evaluation reports, beside what produced them."""
  wins = [outcome for outcome in outcomes if outcome.success]
  named = sorted({name for outcome in outcomes for name in outcome.labels})
  n = len(outcomes)
  return {
    "revision": revision(env),
    "env": env.describe(),
    "n_episodes": n,
    "time_limit": time_limit,
    "success_rate": len(wins) / n,
    "progress": float(np.mean([outcome.progress for outcome in outcomes])),
    "unstable_rate": sum(outcome.unstable for outcome in outcomes) / n,
    **{
      f"{name}_rate": sum(outcome.labels[name] for outcome in outcomes) / n
      for name in named
    },
    "time_to_success": (
      float(np.mean([outcome.time for outcome in wins])) if wins else float("nan")
    ),
    "tracking_error": float(
      np.median([outcome.tracking_error for outcome in outcomes])
    ),
  }


def revision(env: Env[Any]) -> str:
  """A hash of what decides an env's answer to a seed: every number its model
  compiled to, what it was built from, the code that runs it, and MuJoCo."""
  digest = hashlib.sha256()
  model = env.model
  for name in sorted(dir(model)):
    value = getattr(model, name)
    if isinstance(value, np.ndarray) and not name.endswith("_pathadr"):
      digest.update(name.encode())
      digest.update(np.ascontiguousarray(value).tobytes())
  digest.update(json.dumps(env.describe(), sort_keys=True).encode())
  package = Path(specialist_sim.__file__).parent
  for path in sorted(package.rglob("*.py")):
    digest.update(path.relative_to(package).as_posix().encode())
    digest.update(path.read_bytes())
  for kind in (type(env.task), env.cfg.embodiment):
    digest.update(inspect.getsource(sys.modules[kind.__module__]).encode())
  digest.update(mujoco.__version__.encode())
  return digest.hexdigest()[:12]
