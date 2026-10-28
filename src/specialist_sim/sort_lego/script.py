"""The LEGO sorting demonstration, the fallback to teleoperation: one brick at a
time, taken from above, carried and let go over the bin of its color.

A grasp is the operator's at the 0927 takes: fingers along the brick's long
side, pitched ``PITCH`` down. With two hands, the left takes blue and the right
red, turn by turn. The highest brick goes first, then the nearest. A brick not
held once lifted is tried again, ``ATTEMPTS`` times in all. The script reads
every brick's pose off the scene, which no policy is given.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from specialist_sim import rotation
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.scripted import ScriptedPolicy, ScriptedPolicyCfg, Stage, Waypoint
from specialist_sim.sort_lego import initial, scene, task
from specialist_sim.sort_lego.assets import GRASP
from specialist_sim.sort_lego.env import SortLegoEnv

__all__ = ["ATTEMPTS", "Script", "drop_pose", "grasp_pose", "held", "scripted_policy"]

PITCH = np.radians(48.0)
"""The fingers below the horizontal at a grasp: the 0927 grasps lie between 42
and 53 degrees."""
OPEN = 0.3
SHUT = -0.15
"""Openings, radians: open before a grasp, as the takes' hands were at a
cycle's start, and sent past where the pads meet (-0.159), so the jaws squeeze
what they close on."""
ABOVE = 0.06
"""How far over a brick a hand lines up before going down to it, meters."""
LIFT = 0.08
"""How far a closed hand rises before it carries, meters."""
DROP = 0.07
"""How far over a bin's rim a brick is let go, meters."""
HELD = 0.015
"""How far from the grasp point a brick may lie and count as held, meters."""
ATTEMPTS = 3
ARCH = 0.05
"""How high a carry rises over the straight line to the bin, meters."""


def _stages(hand: int) -> tuple[Stage, ...]:
  return (
    Stage(hand, "over", window=(0.0, 0.5)),
    Stage(hand, "down", turns=False),
    Stage(hand, "close", turns=False),
    Stage(hand, "lift", loaded=True, turns=False),
    Stage(hand, "carry", loaded=True),
    Stage(hand, "release", loaded=True, turns=False),
    Stage(hand, "back"),
  )


class Script:
  """The sorting rules for the driven hands, over the bricks ``active`` names."""

  def __init__(
    self, driven: tuple[str, ...], colors: tuple[str, ...], active: BoolArray
  ):
    self.driven, self.colors = driven, colors
    self.active = active.copy()
    self.attempts = np.zeros(len(active), np.int64)
    self.reach: tuple[Float64Array, Float64Array] = (
      np.array([scene.BIN_NEAR_FACE - scene.BIN_SIZE[0] - 0.1, -0.6, scene.DESK_TOP]),
      np.array([scene.TRAY_CENTER[0] + 0.35, 0.6, scene.DESK_TOP + 0.5]),
    )
    self._hand = 0
    self._turn = 0
    self._brick: int | None = None

  @property
  def stages(self) -> tuple[Stage, ...]:
    return _stages(self._hand)

  def owner(self, brick: int) -> int | None:
    """The driven hand that takes ``brick``: its color's side, or the one hand
    there is."""
    if len(self.driven) == 1:
      return 0
    slot = "left_gripper" if self.colors[brick] == "blue" else "right_gripper"
    return self.driven.index(slot) if slot in self.driven else None

  def waiting(self, bricks: Float64Array) -> list[int]:
    """The bricks still to sort: on the tray, owned, and not given up on."""
    return [
      brick
      for brick in map(int, np.flatnonzero(self.active))
      if self.owner(brick) is not None
      and self.attempts[brick] < ATTEMPTS
      and task.on_tray(bricks[brick, :3])
    ]

  def opens(self, hands: Float64Array, bricks: Float64Array) -> int:
    """Take turns: the next hand with a brick of its own takes its highest,
    then its nearest. With nothing left, the cycle stands still."""
    waiting = self.waiting(bricks)
    self._brick = None
    for offset in range(len(self.driven)):
      hand = (self._turn + offset) % len(self.driven)
      mine = [brick for brick in waiting if self.owner(brick) == hand]
      if mine:
        reach = hands[hand, :2]
        self._brick = min(
          mine,
          key=lambda brick: (
            -round(float(bricks[brick, 2]) / scene.BRICK[2]),
            float(np.linalg.norm(bricks[brick, :2] - reach)),
          ),
        )
        self._hand, self._turn = hand, (hand + 1) % len(self.driven)
        self.attempts[self._brick] += 1
        break
    return 0

  def finished(self, bricks: Float64Array) -> bool:
    return not self.waiting(bricks)

  def arrived(
    self, stage: Stage, elapsed: float, hands: Float64Array, bricks: Float64Array
  ) -> bool | None:
    return None

  def waypoint(
    self, stage: Stage, hands: Float64Array, pose: Float64Array, bricks: Float64Array
  ) -> Waypoint:
    """Where this stage sends the hand. Without a brick to sort, or with the
    brick not held once lifted, the hand goes home instead."""
    hand, slot = stage.hand, self.driven[stage.hand]
    home = initial.home(slot)
    going_home = Waypoint(home[:3], OPEN, rotation.to_matrix(home[3:7]))
    if self._brick is None:
      return replace(
        going_home, goal=pose[hand, :3].copy(), opening=float(pose[hand, 7])
      )
    brick = self._brick
    grasp = grasp_pose(bricks[brick], slot)
    facing = rotation.to_matrix(grasp[3:7])
    if stage.name == "over":
      return Waypoint(grasp[:3] + np.array([0.0, 0.0, ABOVE]), OPEN, facing, bulge=0.02)
    if stage.name == "down":
      return Waypoint(grasp[:3], OPEN, facing)
    if stage.name == "close":
      return Waypoint(grasp[:3], SHUT, facing)
    if stage.name == "lift":
      return Waypoint(pose[hand, :3] + np.array([0.0, 0.0, LIFT]), SHUT, facing)
    holding = held(hands[hand], bricks[brick])
    if stage.name == "carry":
      if not holding:
        return going_home
      drop = drop_pose(self.colors[brick])
      return Waypoint(drop[:3], SHUT, rotation.to_matrix(drop[3:7]), bulge=ARCH)
    if stage.name == "release":
      return Waypoint(pose[hand, :3].copy(), OPEN, rotation.to_matrix(pose[hand, 3:7]))
    return going_home


def grasp_pose(brick: Float64Array, slot: str) -> Float64Array:
  """The hand state that holds ``brick`` ``(7,)`` between its pads, ``(7,)``:
  fingers along its long side, pointing away from the hand's home."""
  long_side = rotation.to_matrix(brick[3:7])[:, 0]
  along = np.array([long_side[0], long_side[1], 0.0]) / np.linalg.norm(long_side[:2])
  if np.dot(along[:2], brick[:2] - initial.home(slot)[:2]) < 0.0:
    along = -along
  quaternion = rotation.pointing(
    np.cos(PITCH) * along - np.array([0.0, 0.0, np.sin(PITCH)])
  )
  position = brick[:3] - rotation.rotate(quaternion, np.asarray(GRASP))
  return np.array([*position, *quaternion])


def drop_pose(color: str) -> Float64Array:
  """The hand state that holds a brick ``DROP`` over the middle of ``color``'s
  bin, fingers pitched down toward the bins, ``(7,)``."""
  middle = scene.bin_center(color) + np.array([0.0, 0.0, scene.BIN_SIZE[2] + DROP])
  quaternion = rotation.pointing(np.array([-np.cos(PITCH), 0.0, -np.sin(PITCH)]))
  return np.array(
    [*(middle - rotation.rotate(quaternion, np.asarray(GRASP))), *quaternion]
  )


def held(hand: Float64Array, brick: Float64Array) -> bool:
  """Whether ``brick`` lies between the pads of a hand standing at ``hand``."""
  grasp = hand[:3] + rotation.rotate(hand[3:7], np.asarray(GRASP))
  return bool(np.linalg.norm(grasp - brick[:3]) < HELD)


def scripted_policy(
  env: SortLegoEnv,
  env_idx: int,
  rng: np.random.Generator,
  cfg: ScriptedPolicyCfg | None = None,
) -> ScriptedPolicy:
  """The scene's scripted policy for the episode env ``env_idx`` holds,
  drawing from ``rng``. It reads every brick's pose, ``env.bricks(env_idx)``,
  off the scene, and cycles until no brick is left to sort."""
  layout = env.layouts[env_idx]
  assert layout is not None
  clock = env.cfg.clock
  return ScriptedPolicy(
    Script(env.cfg.driven, env.cfg.colors, layout.active),
    rng,
    clock.action_dt,
    clock.n_action_steps,
    cfg or ScriptedPolicyCfg(length=(0.0, 0.0), rescue=600.0),
  )
