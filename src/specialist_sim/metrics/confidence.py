"""How sure a number is, which is half of what a number means.

A success rate quoted without an interval is the single most common way a robot
evaluation overstates itself. Twenty episodes at 60% is consistent with anything
from about 39% to 78%, so two policies a few points apart over twenty episodes
have not been distinguished, and reporting them as though they had is how a
benchmark accumulates gains that are not there.

Two kinds of interval, because there are two kinds of number:

**A rate is a proportion**, and gets a Wilson score interval. Wilson rather than
the exact Clopper-Pearson one: it is tighter at these sample sizes and it stays
sensible at 0 and 1, where the normal approximation collapses to zero width and
claims certainty from a handful of episodes.

**Everything else is a distribution**, and gets a percentile bootstrap. It makes
no assumption about shape, which matters because tracking error and lift are not
remotely normal. The resampling is seeded, so a summary of the same episodes is
the same summary twice.

Neither interval says anything about how a policy will do on a real robot. They
describe sampling error *within this simulator* -- how much of the number is the
handful of episodes drawn. The gap between this rig and a real one is a bias, not
a variance, and no amount of simulated episodes shrinks it.

Delete this module and its two calls in :mod:`specialist_sim.metrics.summary` to
go back to bare point estimates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np

__all__ = ["CONFIDENCE", "RESAMPLES", "bootstrap", "wilson"]

CONFIDENCE = 0.95
"""The two-sided level every interval here is quoted at."""
RESAMPLES = 2000
"""Bootstrap resamples. Enough that the interval is stable to about a percent of
its own width, and cheap at the episode counts an evaluation runs."""
Z = 1.959963984540054
"""The normal quantile for :data:`CONFIDENCE`, written out so nothing here needs
scipy."""


def wilson(wins: int, trials: int) -> tuple[float, float]:
  """A Wilson score interval on a proportion.

  Args:
    wins: How many episodes had the outcome.
    trials: How many there were.

  Returns:
    The low and high end of the interval. ``(nan, nan)`` for no trials at all.
  """
  if trials <= 0:
    return (float("nan"), float("nan"))
  z2 = Z * Z
  middle = (wins + z2 / 2.0) / (trials + z2)
  spread = Z / (trials + z2) * np.sqrt(wins * (trials - wins) / trials + z2 / 4.0)
  # At no wins and at all wins the two terms are equal in exact arithmetic and
  # differ by rounding in this one, which would put the bound a floating hair off
  # the end of the range it is quoted in. Both ends are exact, so they are said
  # rather than computed.
  low = 0.0 if wins == 0 else max(0.0, middle - spread)
  high = 1.0 if wins == trials else min(1.0, middle + spread)
  return (float(low), float(high))


def bootstrap(
  values: Sequence[float],
  statistic: Callable[[np.ndarray], float] = lambda sample: float(np.median(sample)),
  seed: int = 0,
) -> tuple[float, float]:
  """A percentile bootstrap interval on any statistic of a sample.

  Args:
    values: The sample. NaNs are dropped first, so an episode that reported
      nothing does not drag the interval to NaN with it.
    statistic: What is being estimated. The median, by default.
    seed: Fixes the resampling, so the same episodes summarise identically.

  Returns:
    The low and high end of the interval, or ``(nan, nan)`` for fewer than two
    values -- one value is a point, not a distribution.
  """
  kept = np.asarray([value for value in values if not np.isnan(value)], np.float64)
  if len(kept) < 2:
    return (float("nan"), float("nan"))
  draw = np.random.default_rng(seed)
  picks = draw.integers(0, len(kept), size=(RESAMPLES, len(kept)))
  spread = np.array([statistic(kept[row]) for row in picks])
  tail = (1.0 - CONFIDENCE) / 2.0
  low, high = np.percentile(spread, [100.0 * tail, 100.0 * (1.0 - tail)])
  return (float(low), float(high))
