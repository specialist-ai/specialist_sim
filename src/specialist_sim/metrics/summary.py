"""Many episodes, as the numbers a run is judged on.

Where :mod:`specialist_sim.metrics.episode` says what one episode did, this says
what a set of them did, and it fixes the choices that otherwise get made
differently every time someone writes a table.

**Every number comes with an interval.** A success rate over twenty episodes is
worth about plus or minus twenty points, so two policies four points apart over
twenty episodes have not been distinguished. Reporting the point estimate alone
is how an evaluation accumulates improvements that were never there, and it is
the single most common defect in published robot evaluations. Rates get a Wilson
score interval and distributions get a percentile bootstrap; see
:mod:`specialist_sim.metrics.confidence`.

**A rate is over every episode, and a conditional mean is over the episodes it
applies to.** ``success_rate`` counts the whole set; ``steps_to_success``
averages only the wins, because averaging a failure's step count into it measures
how long the budget is rather than how quick the policy is. Every conditional
number is reported with the ``_n`` it was taken over, because a mean over two
episodes and a mean over twenty are not the same claim. Where there is nothing to
average over the answer is a NaN rather than a zero.

**A distribution is reported by its median, and by its tail where the tail is the
risk.** Tracking and lift are medians: one episode where a hand was sent
somewhere it could not go should not move the number describing the other
nineteen. ``tracking_p90`` is there because for tracking the tail is the
interesting part -- a rig that is usually accurate and occasionally 3 cm out is
a different rig from one that is always 3 mm out.

``truncated_rate`` is the share the environment's own horizon ended and
``stopped_rate`` the share the caller ended, which are different facts about a
run: the first is a task the policy could not finish in the time the task allows,
the second is an evaluation that ran out of patience. Neither is folded into the
other.

**The stage rates are the failure breakdown.** ``reached_rate`` down to
``placed_rate`` is the share of episodes in which each thing actually happened,
so the biggest drop along the sequence is where the policy is failing.
``placed_rate`` is ``success_rate`` by construction, kept because seeing the
sequence end where the success rate is makes the sequence readable.

They usually fall from left to right and they are **not** forced to. Each is the
share of episodes where that thing happened, not the share that got at least that
far, because in this task the stages are not strictly ordered: a cube can be
knocked into the air by the back of a gripper, giving ``lifted`` without
``touched``. Where the sequence does not fall, that is the finding -- a policy
scattering the cube rather than picking it up -- and forcing monotonicity would
report grasps that never happened. ``progress`` is the monotone number.

Two things this deliberately does not do. It does not test whether two runs
differ: comparing policies wants a paired test over the per-episode reports, and
the reports are kept so that a caller can run one. And it does not say anything
about a real robot -- these intervals describe sampling error inside this
simulator, and the gap to hardware is a bias no number of simulated episodes
shrinks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from specialist_sim.metrics.confidence import bootstrap, wilson
from specialist_sim.metrics.episode import STAGES, EpisodeReport

__all__ = ["summarise"]


def summarise(reports: Sequence[EpisodeReport]) -> dict[str, Any]:
  """The numbers a set of episodes is judged on.

  Args:
    reports: One per episode, from :meth:`Episode.report`. Episodes from
      different worlds of one batch and from separate runs are the same thing
      here: a report does not remember which world produced it.

  Returns:
    Plain floats and ints, ready for a logger. Every rate carries ``_low`` and
    ``_high``; every median carries them too; every conditional mean carries an
    ``_n``. Empty in gives every field a NaN rather than an error, because a run
    that has scored nothing yet is a normal state and not a bug.

  Raises:
    ValueError: An episode has not ended, so its numbers are not final; or two
      episodes report the same seed, which means one episode has been counted
      twice and every rate below is wrong.
  """
  unfinished = [one.seed for one in reports if one.ended == "running"]
  if unfinished:
    raise ValueError(f"{len(unfinished)} episodes have not ended: seeds {unfinished}")
  _distinct(reports)
  wins = [one for one in reports if one.success]
  touched = [one for one in reports if one.first_contact >= 0]
  out: dict[str, Any] = {"episodes": len(reports)}
  out.update(_rate("success_rate", [one.success for one in reports]))
  out.update(_rate("pick_rate", [one.picked for one in reports]))
  out.update(_rate("dropped_rate", [one.dropped for one in reports]))
  out.update(_rate("truncated_rate", [one.ended == "truncated" for one in reports]))
  out.update(
    _rate(
      "stopped_rate",
      [one.ended not in ("success", "dropped", "truncated") for one in reports],
    )
  )
  for stage in STAGES:
    out.update(_rate(f"{stage}_rate", [stage in one.stages for one in reports]))
  out.update(_middle("progress", [one.progress for one in reports]))
  out.update(_average("steps_to_success", [one.steps_to_success for one in wins]))
  out.update(_average("steps", [one.steps for one in reports]))
  out.update(_average("reward", [one.reward for one in reports]))
  out.update(_average("first_contact", [one.first_contact for one in touched]))
  out.update(_middle("lift", [one.lift for one in reports]))
  out.update(_middle("closest_approach", [one.closest_approach for one in reports]))
  out.update(_middle("held_steps", [one.held_steps for one in reports]))
  out.update(_middle("carried_steps", [one.carried_steps for one in reports]))
  out.update(_middle("path_length", [one.path_length for one in reports]))
  out.update(_middle("smoothness", [one.smoothness for one in reports]))
  out.update(_middle("clipped", [one.clipped for one in reports]))
  out.update(_middle("tracking", [one.tracking for one in reports]))
  out.update(_middle("tracking_rotation", [one.tracking_rotation for one in reports]))
  out["tracking_p90"] = _tail([one.tracking for one in reports])
  out["penetration"] = max((one.penetration for one in reports), default=float("nan"))
  out["image_age"] = max((one.image_age for one in reports), default=0)
  return out


def _distinct(reports: Sequence[EpisodeReport]) -> None:
  """Refuse a set that counts one episode twice.

  A seed names an episode, so two reports sharing one are either the same episode
  scored twice or a protocol that did not vary what it meant to vary. Both make
  every rate above wrong, and both are easy to do by accident when a batch of
  worlds is reset from one seed.
  """
  seeds = [one.seed for one in reports if one.seed is not None]
  repeated = sorted({seed for seed in seeds if seeds.count(seed) > 1})
  if repeated:
    raise ValueError(
      f"{len(seeds) - len(set(seeds))} episodes repeat a seed, so they are not "
      f"independent: {repeated}"
    )


def _rate(name: str, flags: Sequence[bool]) -> dict[str, float]:
  """A proportion, with a Wilson score interval."""
  if not flags:
    return {
      name: float("nan"),
      f"{name}_low": float("nan"),
      f"{name}_high": float("nan"),
    }
  low, high = wilson(int(sum(flags)), len(flags))
  return {name: float(np.mean(flags)), f"{name}_low": low, f"{name}_high": high}


def _middle(name: str, values: Sequence[float]) -> dict[str, float]:
  """A median, with a bootstrap interval, over the episodes that reported one."""
  kept = [float(value) for value in values if not np.isnan(value)]
  if not kept:
    return {
      name: float("nan"),
      f"{name}_low": float("nan"),
      f"{name}_high": float("nan"),
    }
  low, high = bootstrap(kept)
  return {name: float(np.median(kept)), f"{name}_low": low, f"{name}_high": high}


def _average(name: str, values: Sequence[float]) -> dict[str, Any]:
  """A conditional mean, with its interval and the count it was taken over."""
  kept = [float(value) for value in values if not np.isnan(value)]
  if not kept:
    return {
      name: float("nan"),
      f"{name}_low": float("nan"),
      f"{name}_high": float("nan"),
      f"{name}_n": 0,
    }
  low, high = bootstrap(kept, statistic=lambda sample: float(np.mean(sample)))
  return {
    name: float(np.mean(kept)),
    f"{name}_low": low,
    f"{name}_high": high,
    f"{name}_n": len(kept),
  }


def _tail(values: Sequence[float]) -> float:
  kept = [value for value in values if not np.isnan(value)]
  return float(np.percentile(kept, 90)) if kept else float("nan")
