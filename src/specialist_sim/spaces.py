"""What an action and an observation look like.

An action is a trajectory: ``HORIZON`` poses spaced ``1 / ACTION_HZ`` apart,
naming where the hands should be from now to ``HORIZON / ACTION_HZ`` ahead. A
control step tracks the poses that fall inside it and discards the rest, so the
trajectory reaches past what one step executes and is finer than one step
resolves.

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
  "GRIPPERS",
  "HORIZON",
  "IMAGE_SHAPE",
  "MASK_RADIUS",
  "PER_HAND",
  "STATE_HIGH",
  "STATE_LOW",
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
HORIZON = 16
"""How many poses a trajectory names."""

_HAND_LOW = (0.25, -0.55, -0.335, *(-1.0,) * 4, 0.0)
_HAND_HIGH = (0.85, 0.65, 0.25, *(1.0,) * 4, 1.0)
"""A hand reaches across the table and no lower than its top. A unit
quaternion's components already lie inside the bound the rotation carries."""

PER_HAND = len(_HAND_LOW)
STATE_LOW: Float32Array = np.array(_HAND_LOW * len(GRIPPERS), dtype=np.float32)
STATE_HIGH: Float32Array = np.array(_HAND_HIGH * len(GRIPPERS), dtype=np.float32)
ACTION_LOW: Float32Array = np.tile(STATE_LOW, (HORIZON, 1))
ACTION_HIGH: Float32Array = np.tile(STATE_HIGH, (HORIZON, 1))


def gym_spaces(image_shape: tuple[int, int, int] = IMAGE_SHAPE) -> tuple[Any, Any]:
  """The observation and action spaces as Gymnasium objects."""
  import gymnasium as gym

  observation = gym.spaces.Dict(
    {
      **{name: gym.spaces.Box(0, 255, image_shape, np.uint8) for name in GRIPPERS},
      "state": gym.spaces.Box(STATE_LOW, STATE_HIGH, dtype=np.float32),
    }
  )
  return observation, gym.spaces.Box(ACTION_LOW, ACTION_HIGH, dtype=np.float32)
