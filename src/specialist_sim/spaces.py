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

from specialist_sim.arrays import Float32Array

__all__ = [
  "ACTION_HIGH",
  "ACTION_HZ",
  "ACTION_LOW",
  "FINGER_HIGH",
  "FINGER_LOW",
  "GRIPPERS",
  "HORIZON",
  "IMAGE_SHAPE",
  "MASK_RADIUS",
  "PER_HAND",
  "STATE_HIGH",
  "STATE_LOW",
  "F_STEP",
  "gym_spaces",
]

GRIPPERS = ("left_gripper", "right_gripper")
"""The hands, named for the operator's side each one is. Every body a gripper
owns carries its name as a prefix, and so does the image it sees."""
IMAGE_SHAPE = (240, 240, 3)
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
HORIZON = 16
"""How many poses a trajectory names."""

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


def gym_spaces(
  image_shape: tuple[int, int, int] = IMAGE_SHAPE, rows: int = 10
) -> tuple[Any, Any]:
  """The observation and action spaces as Gymnasium objects.

  An observation carries every row a control step spanned, and says which row
  each one is. The rows are the robot's own clock: exact, and current as of the
  boundary the step returns at. The image is on the camera's clock, which runs
  behind the robot's, so it is quarantined behind ``image_step``, the row it was
  captured at. Nothing here reports how far behind that is; a reader subtracts.
  """
  import gymnasium as gym

  low, high = STATE_LOW[:PER_HAND], STATE_HIGH[:PER_HAND]
  hand = gym.spaces.Dict(
    {
      "pose": gym.spaces.Box(
        np.tile(low[:7], (rows, 1)), np.tile(high[:7], (rows, 1)), dtype=np.float32
      ),
      "opening": gym.spaces.Box(low[7], high[7], (rows,), np.float32),
      "image": gym.spaces.Box(0, 255, image_shape, np.uint8),
      "image_step": gym.spaces.Box(-(2**31), 2**31 - 1, (), np.int64),
    }
  )
  observation = gym.spaces.Dict(
    {
      "step": gym.spaces.Box(-(2**31), 2**31 - 1, (rows,), np.int64),
      **{slot: hand for slot in GRIPPERS},
    }
  )
  return observation, gym.spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)
