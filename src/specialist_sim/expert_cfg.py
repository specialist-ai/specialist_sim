"""What a demonstration is like, apart from the machinery that performs it.

Every number here is a style choice measured off the human recordings in
corpus_0821 or sized to this scene; the driver in ``expert`` consumes it, and
nothing here imports the driver, so a sweep or a training config can read and
build one without standing an environment up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ExpertCfg"]


@dataclass(frozen=True)
class ExpertCfg:
  across: tuple[float, float] = (0.35, 1.5)
  """How long a move takes: a fixed part plus a share of the distance. A longer
  reach is carried by speed rather than by time."""
  speed: float = 0.8
  """The fastest a hand travels, whatever the time its distance asks for."""
  kick: tuple[float, float] = (0.35, 0.10)
  """The speed a strike sweeps at, and how far that carries the cube."""
  wrist: float = np.radians(14.0)
  """How far a hand turns about the vertical from one stage to the next, which
  is the spread the human's grasps are turned over."""
  squaring: float = 0.5
  """The share of a move that ends at the cube which the wrist turns over. The
  pads stand off the hand, so a wrist still turning swings them sideways; past
  this the hand only travels, and they come at the cube straight. Every other
  move turns the whole way, so the hand rotates as it goes."""
  patch: tuple[float, float] = (0.035, 0.020)
  """How far a kick scatters where the cube lands, across the line it is driven
  along and along it. The two are in the proportion the human's picks scatter
  in; the size is what leaves the cube in reach."""
  tempo: tuple[float, float] = (0.85, 1.15)
  """How fast one episode runs against another."""
  reach: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (0.25, -0.55, -0.335),
    (0.85, 0.65, 0.25),
  )
  """The corner to corner box a hand is aimed inside: across the table, and no
  lower than its top. A goal past it would buy a stage the time its distance
  asks for and leave the hand nowhere the cube is."""
  hurry: float = 0.12
  """How much an episode quickens between its first cycle and its last."""
  flow: float = 0.10
  """How fast a hand is still going as one stage hands over to the next, as a
  share of its top speed. A hand that takes the cube and lets it go on the move
  never comes to a stop over it."""
  finger_rate: float = 4.0
  """The fastest the fingers move when nothing between them holds them back."""
  let_go: float = 0.7
  """The fastest they open on a cube they still hold. The servo presses on it,
  so opening quickly fires it out from between the pads."""
  freed: float = 0.06
  """How much further than holding the fingers open to let the cube go. The
  rest of the opening happens on the way out, with nothing between them."""
  bite: float = 0.02
  """How far inside touching the fingers close on the cube. The servo arrives
  where it is sent, so how hard the cube is held follows from where the pads
  come to rest against it rather than from this."""
  park: tuple[float, float] = (0.20, 0.25)
  """How far from the tape a hand stands once it is done. A gripper's own parts
  reach 0.12 m from the point its command names, so two of them need this much
  between them for the idle one to be clear of the working one."""
  stray: float = np.radians(15.0)
  """How far around the tape a parked spot swings, so a hand comes to rest
  somewhere new each time rather than tracing one pose over and over."""
  lift: tuple[float, float] = (0.126, 0.006)
  """How high a carried cube rides over its travel: a share of the distance
  plus a fixed part. Measured off the recordings, where a hand lifts about a
  fifth of what it crosses."""
  skim: float = 0.015
  """How high an empty hand rises on its way. A hand with nothing in it skims
  the table rather than lifting, crossing and setting down."""
  arch: float = 0.06
  """The most a hand rises, however far it has to go."""
  swing: float = 0.07
  """How far either side of the cube a kick starts and ends."""
  clear: float = 0.40
  """An opening that clears the cube."""
  linger: float = 0.4
  """How long it stands clear at the end, so the cube is seen to be let go."""
  pace: tuple[float, float] = (1.0, 1.3)
  """How far a cycle stretches the time each of its stages takes. A cycle never
  runs shorter than its moves ask for, so ``speed`` stays a ceiling."""
  length: tuple[float, float] = (80.0, 95.0)
  """How long an episode runs before it stops starting cycles. It then sees the
  one it is in through, so a recording outlasts this by up to a cycle."""
  rescue: float = 12.0
  """How long past its length an episode keeps cycling to end with the cube on
  the tape, after which it stops wherever the cube lies."""
