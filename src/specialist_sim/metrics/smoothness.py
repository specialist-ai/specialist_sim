"""How fluent a motion was, which a success rate cannot see.

Two policies can place the cube equally often and move nothing alike -- one
gliding, one juddering to the same place through a dozen corrections. The
outcome metrics score those identically, and on real hardware they are not the
same policy at all: the juddering one wears gearboxes, trips torque limits and
transfers worse.

The measure is **SPARC**, the spectral arc length of the hand's speed profile
(Balasubramanian et al.). It is the arc length of the normalised Fourier
magnitude spectrum of speed, up to a cutoff. A smooth reach has a compact
spectrum and a short arc; every correction adds high-frequency content and
lengthens it. Values are negative and **less negative is smoother**: about -1.4
for a single clean reach, and falling as intermittency grows.

Why this and not jerk. Jerk is a third derivative, so on a sampled trajectory it
is mostly differentiation noise, and every dimensionless form of it needs the
movement to start and end at rest -- which an episode of continuous pick and
place does not. SPARC needs neither: it is dimensionless by construction, robust
to noise, and defined for a speed profile that never returns to zero. Of the
published smoothness measures it and log dimensionless jerk are the two with
excellent test-retest reliability; only SPARC survives our boundary conditions.

Delete this module, the ``smoothness`` field, and the buffer that feeds it in
:mod:`specialist_sim.metrics.episode` to drop the measure.
"""

from __future__ import annotations

import numpy as np

from specialist_sim.arrays import Float64Array

__all__ = ["CUTOFF_HZ", "THRESHOLD", "sparc"]

CUTOFF_HZ = 10.0
"""The highest frequency the arc is measured over. Above this a hand is not
moving, it is being sampled: the trajectory rows come at 100 Hz and the rig's own
tracking bandwidth is well under this."""
THRESHOLD = 0.05
"""Where the spectrum is cut, as a fraction of its own peak. Trims the noise
floor beyond the last real frequency content, so a longer quiet tail does not
lengthen the arc."""
PAD = 4
"""How far the profile is zero-padded, as powers of two beyond its own length.
Interpolates the spectrum finely enough that the arc length stops depending on
how many rows the episode happened to have."""


def sparc(speed: Float64Array, rate: float) -> float:
  """The spectral arc length of a speed profile. Less negative is smoother.

  Args:
    speed: ``(rows,)`` speed magnitude, in any unit -- the spectrum is
      normalised, so the measure is dimensionless and scale-free.
    rate: How many rows a second, in Hz.

  Returns:
    The arc length, always negative. NaN for a profile too short to have a
    spectrum, or one that never moved at all.
  """
  profile = np.asarray(speed, np.float64)
  if len(profile) < 4 or not np.isfinite(profile).all():
    return float("nan")
  points = int(2 ** (np.ceil(np.log2(len(profile))) + PAD))
  spectrum = np.abs(np.fft.rfft(profile, points))
  peak = spectrum.max()
  if peak <= 0.0:
    return float("nan")
  spectrum = spectrum / peak
  frequency = np.fft.rfftfreq(points, 1.0 / rate)
  within = frequency <= CUTOFF_HZ
  frequency, spectrum = frequency[within], spectrum[within]
  # Cut the tail past the last frequency carrying real content, so the noise
  # floor beyond a movement does not count as part of it.
  loud = np.flatnonzero(spectrum >= THRESHOLD)
  if len(loud) < 2:
    return float("nan")
  frequency = frequency[loud[0] : loud[-1] + 1]
  spectrum = spectrum[loud[0] : loud[-1] + 1]
  span = frequency[-1] - frequency[0]
  if span <= 0.0:
    return float("nan")
  steps = np.sqrt((np.diff(frequency) / span) ** 2 + np.diff(spectrum) ** 2)
  return float(-steps.sum())
