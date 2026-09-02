"""What an action and an observation look like.

An action is a trajectory: ``HORIZON`` poses spaced ``1 / ACTION_HZ`` apart,
naming where the hands should be from now to ``HORIZON / ACTION_HZ`` ahead. A
control step tracks the poses that fall inside it and discards the rest, so the
trajectory reaches past what one step executes and is finer than one step
resolves.

A hand's opening, in an action and in an observation alike, is the coupled
finger angle in radians, signed as the left finger turns. That is the number a
corpus records, so nothing converts between what is stored and what is
commanded.

Shapes and bounds as plain values, and Gymnasium space objects on request.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from specialist_sim.arrays import Float32Array, Float64Array, Int64Array

__all__ = [
  "ROWS_PER_POSE",
  "action_bounds",
  "pose_rows",
  "ACTION_HIGH",
  "ACTION_HZ",
  "ACTION_LOW",
  "FINGER_HIGH",
  "FINGER_LOW",
  "FINGER_OPEN",
  "FINGER_SHUT",
  "GRIPPERS",
  "HORIZON",
  "MASK_RADIUS",
  "PER_HAND",
  "STATE_HIGH",
  "STATE_LOW",
  "F_STEP",
  "gripper_from_opening",
  "gym_spaces",
  "opening_from_gripper",
]

GRIPPERS = ("left_gripper", "right_gripper")
"""The hands, named for the operator's side each one is. Every body a gripper
owns carries its name as a prefix, and so does the image it sees."""
MASK_RADIUS = 0.5
"""How much of the frame's width the observation keeps, as a radius from its
centre. The recorded observations are blanked outside the same disc, and the
lens model stays invertible a little past it."""

ACTION_HZ = 50.0
"""How often a trajectory names a pose."""
F_STEP = 100.0
"""How often the step timeline ticks. A row is one of its ticks, and it is the
rate a corpus records a hand at."""

FINGER_LOW = -0.261799
FINGER_HIGH = 0.698132
"""How far the finger joint travels, which is what an opening names."""
FINGER_SHUT = np.radians(-9.1175)
"""The angle at which the pads meet, which a gripper command of zero asks for."""
FINGER_OPEN = np.radians(40.0)
"""The joint's open limit, which a gripper command of one asks for."""


def gripper_from_opening(opening: float) -> float:
  """The joint angle an opening asks for, saturating at shut and open."""
  span = float(np.clip(opening, 0.0, 1.0))
  return float(FINGER_SHUT + span * (FINGER_OPEN - FINGER_SHUT))


def opening_from_gripper(angle: float) -> float:
  """The opening an angle stands for, saturating at shut and open."""
  return float(np.clip((angle - FINGER_SHUT) / (FINGER_OPEN - FINGER_SHUT), 0.0, 1.0))


HORIZON = 16
"""How many poses a trajectory names."""

ROWS_PER_POSE = round(F_STEP / ACTION_HZ)
"""How many rows of the step timeline one pose of a trajectory spans."""


def pose_rows(anchor: int, poses: int = HORIZON) -> Int64Array:
  """The rows a trajectory planned at ``anchor`` names: pose ``i`` is reached
  at the end of the ``i``-th span after the anchor."""
  return anchor + ROWS_PER_POSE * np.arange(1, poses + 1, dtype=np.int64)


_HAND_LOW = (0.25, -0.55, -0.335, *(-1.0,) * 4, FINGER_LOW)
_HAND_HIGH = (0.85, 0.65, 0.25, *(1.0,) * 4, FINGER_HIGH)
"""A hand reaches across the table and no lower than its top. A unit
quaternion's components already lie inside the bound the rotation carries, and
the fingers reach as far as their joint does."""

PER_HAND = len(_HAND_LOW)
STATE_LOW: Float32Array = np.array(_HAND_LOW * len(GRIPPERS), dtype=np.float32)
STATE_HIGH: Float32Array = np.array(_HAND_HIGH * len(GRIPPERS), dtype=np.float32)
ACTION_LOW: Float32Array = np.tile(STATE_LOW, (HORIZON, 1))
ACTION_HIGH: Float32Array = np.tile(STATE_HIGH, (HORIZON, 1))


def action_bounds(hands: int) -> tuple[Float64Array, Float64Array]:
  """The action box for a policy driving ``hands`` of the rig's grippers."""
  low = np.tile(np.asarray(_HAND_LOW, np.float64), (HORIZON, hands))
  high = np.tile(np.asarray(_HAND_HIGH, np.float64), (HORIZON, hands))
  return low, high


def gym_spaces(
  image_shape: tuple[int, int, int] = (240, 240, 3),
  rows: int = 10,
  driven: tuple[str, ...] = GRIPPERS,
  lenses: tuple[str, ...] = GRIPPERS,
) -> tuple[Any, Any]:
  """The observation and action spaces as Gymnasium objects.

  An observation carries every row a control step spanned, and says which row
  each one is. The rows are the robot's own clock: exact, and current as of the
  boundary the step returns at. The image is on the camera's clock, which runs
  behind the robot's, so it is quarantined behind ``image_step``, the row it was
  captured at. A driven hand carries its rows; a lens carries its image; a
  slot that is both carries both.
  """
  import gymnasium as gym

  low, high = STATE_LOW[:PER_HAND], STATE_HIGH[:PER_HAND]
  rows_of = {
    "pose": gym.spaces.Box(
      np.tile(low[:7], (rows, 1)), np.tile(high[:7], (rows, 1)), dtype=np.float32
    ),
    "opening": gym.spaces.Box(low[7], high[7], (rows,), np.float32),
  }
  image_of = {
    "image": gym.spaces.Box(0, 255, image_shape, np.uint8),
    "image_step": gym.spaces.Box(-(2**31), 2**31 - 1, (), np.int64),
  }
  slots: dict[str, Any] = {}
  for slot in GRIPPERS:
    fields: dict[str, Any] = {}
    if slot in driven:
      fields.update(rows_of)
    if slot in lenses:
      fields.update(image_of)
    if fields:
      slots[slot] = gym.spaces.Dict(fields)
  observation = gym.spaces.Dict(
    {"step": gym.spaces.Box(-(2**31), 2**31 - 1, (rows,), np.int64), **slots}
  )
  act_low, act_high = action_bounds(len(driven))
  return observation, gym.spaces.Box(act_low, act_high, dtype=np.float32)
