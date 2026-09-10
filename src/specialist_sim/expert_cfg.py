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
  across: tuple[float, float] = (0.18, 0.90)
  """How long a move takes: a fixed part plus a share of the distance. A longer
  reach is carried by speed rather than by time.

  Set against the recordings rather than to taste: the recorded picking hand
  grasps 76 times in a 136 s session, one pick every 1.8 s, where this expert
  managed 17.9 in 92 s -- one every 5.1 s, three times too slow. A cycle is seven
  stages, so 1.8 s of it leaves each about a quarter of a second."""
  speed: float = 1.0
  """The fastest a hand travels, whatever the time its distance asks for."""
  kick: tuple[float, float] = (0.35, 0.16)
  """The speed a strike sweeps at, and how far that carries the cube.

  How far it carries sets how far the picking hand then has to bring it back,
  and that is measured: the recorded hand carries a median 96 mm between a grasp
  and its release, against 59 mm here at 0.10."""
  wrist: float = np.radians(14.0)
  """How far a hand turns about the vertical from one stage to the next, which
  is the spread the human's grasps are turned over."""
  squaring: float = 0.5
  """The share of a move that ends at the cube which the wrist turns over. The
  pads stand off the hand, so a wrist still turning swings them sideways; past
  this the hand only travels, and they come at the cube straight. Every other
  move turns the whole way, so the hand rotates as it goes."""
  patch: tuple[float, float] = (0.045, 0.035)
  """How far a kick scatters where the cube lands, across the line it is driven
  along and along it. The two are in the proportion the human's picks scatter
  in; the size is what leaves the cube in reach.

  Widened to the spread the recordings carry: their carries run 50 mm at the
  tenth percentile to 156 mm at the ninetieth, where a narrower patch gave
  36 to 77."""
  tempo: tuple[float, float] = (0.85, 1.15)
  """How fast one episode runs against another."""
  reach: tuple[tuple[float, float, float], tuple[float, float, float]] = (
    (0.25, -0.55, -0.260),
    (0.85, 0.65, 0.25),
  )
  """The corner to corner box a hand is aimed inside: across the table, and no
  lower than its top. A goal past it would buy a stage the time its distance
  asks for and leave the hand nowhere the cube is."""
  hurry: float = 0.10
  """How much an episode quickens between its first cycle and its last."""
  flow: float = 0.10
  """How fast a hand is still going as one stage hands over to the next, as a
  share of its top speed. A hand that takes the cube and lets it go on the move
  never comes to a stop over it."""
  finger_rate: float = 6.0
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
  come to rest against it rather than from this.

  The recorded fingers close to within 0.010 rad of their own hard stop while
  holding the cube; these stop 0.142 rad short of theirs, because the pads meet a
  rigid cube at ``Scene.pinch`` and the servo is only sent as far as
  ``pinch - bite``. That is the largest single mismatch between the two datasets
  in anything a policy reads -- 1.8 pooled standard deviations, against under 0.3
  for every geometric entry -- so this closes further. Not the whole way: past
  touching the servo presses on the cube and stores the force in the contact,
  and ``let_go`` then has to bleed it off or the fingers fire the cube out."""
  park: tuple[float, float] = (0.20, 0.25)
  """How far from the tape a hand stands once it is done. A gripper's own parts
  reach 0.12 m from the point its command names, so two of them need this much
  between them for the idle one to be clear of the working one."""
  stray: float = np.radians(15.0)
  """How far around the tape a parked spot swings, so a hand comes to rest
  somewhere new each time rather than tracing one pose over and over."""
  lift: tuple[float, float] = (0.150, 0.010)
  """How high a carried cube rides over its travel: a share of the distance
  plus a fixed part. Measured off the recordings, where a hand lifts about a
  fifth of what it crosses."""
  skim: float = 0.015
  """How high an empty hand rises on its way. A hand with nothing in it skims
  the table rather than lifting, crossing and setting down."""
  arch: float = 0.075
  """The most a hand rises, however far it has to go. The recorded hand peaks a
  median 42 mm over the table while carrying, against 33 mm here."""
  swing: float = 0.07
  """How far either side of the cube a kick starts and ends."""
  clear: tuple[float, float] = (0.26, 0.40)
  """An opening that clears the cube, drawn afresh each time it is asked for.

  A single number was the second-worst mismatch in the dataset: an empty
  scripted hand reported one clearance exactly -- 0.40 of finger travel, zero
  spread over 91,575 rows -- where the recorded hand's empty openings run
  continuously from 0.19 to 0.34 of travel about a median of 0.26. A policy
  cannot learn to command a clearance from a constant."""
  linger: float = 0.12
  """How long it stands clear at the end, so the cube is seen to be let go."""
  pace: tuple[float, float] = (1.0, 1.15)
  """How far a cycle stretches the time each of its stages takes. A cycle never
  runs shorter than its moves ask for, so ``speed`` stays a ceiling."""
  length: tuple[float, float] = (80.0, 95.0)
  """How long an episode runs before it stops starting cycles. It then sees the
  one it is in through, so a recording outlasts this by up to a cycle."""
  loiter: tuple[float, float] = (0.05, 0.14)
  """How far an idle hand drifts before it is given somewhere new to be.

  No hand in the recordings holds still: the picking hand's executed step is
  under a millimeter on 0.6% of rows, where a hand told to wait repeats one
  setpoint exactly and reads 54.8%. More than half of what a policy learned from
  this dataset was an instruction not to move. A hand between picks is traveling
  -- off the tape, round to where the cube will be -- so an idle one here keeps
  moving rather than being frozen at its last command."""
  idle_speed: float = 0.22
  """How fast an idle hand drifts, well under ``speed``: it is repositioning
  rather than going anywhere."""
  brief: float = 0.25
  """The least time a stage takes, so a move that barely travels still eases in
  and out of it.

  It is not what limits how often the hand picks, though it looks as if it
  should: seven stages at 0.25 s cannot run faster than 1.75 s, and the
  recordings pick every 1.8 s, so the floor and the target are the same number.
  Lowering it to 0.15 moved the measured rate from 0.31 to 0.30 picks a second
  -- nothing. Stages are not ending on their own clock: ``_arrived`` also waits
  for the plant to come within ``_REACHED`` of the goal, and gives it up to
  ``_DWELL`` to do it, so a commanded move that outruns the controller is
  followed by a wait at a standing setpoint. That wait is both the missing rate
  and most of the motion still missing from the executed target, so the next
  lever is how well the controller tracks, not how fast the demonstration is
  told to move."""
  """How fast an idle hand drifts, well under ``speed``: it is repositioning
  rather than going anywhere."""
  rescue: float = 12.0
  """How long past its length an episode keeps cycling to end with the cube on
  the tape, after which it stops wherever the cube lies."""
