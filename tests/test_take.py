"""Reading a recorded take back as poses the environment can be commanded with."""

from typing import cast

import numpy as np
import pytest

pytest.importorskip("specialist_dataset", reason="the dataset extra is not installed")

from specialist_sim.take import Take, Track, _world_from_room  # noqa: E402


def test_the_room_frame_stands_midway_between_the_anchors() -> None:
  """The origin is the anchors' midpoint, +y runs from the right anchor to the
  left one, +z is up."""
  left, right = np.eye(4), np.eye(4)
  left[:3, 3], right[:3, 3] = [1.0, 0.5, 0.3], [1.0, -0.5, 0.3]
  world_from_room = _world_from_room(left, right)
  assert np.allclose(world_from_room[:3, 3], [1.0, 0.0, 0.3])
  assert np.allclose(world_from_room[:3, :3], np.eye(3))


class _Standing:
  """A take whose room frame is the world frame."""

  room_from_world = np.eye(4)


def test_a_pose_is_read_between_the_samples_it_falls_between() -> None:
  times = np.array([0.0, 1.0])
  poses = np.tile(np.eye(4), (2, 1, 1))
  poses[1, :3, 3] = [1.0, 0.0, 0.0]
  poses[1, :3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
  track = Track(
    hand="right",
    slot="cam01",
    tag_from_lens=np.eye(4),
    times=times,
    poses=poses,
    opening=np.array([0.0, 0.2]),
    ticks=times,
    frames=[b"", b""],
  )
  standing = cast(Take, _Standing())
  ends = Take.poses(standing, track, np.array([0.0, 1.0]))
  assert np.allclose(ends[:, :3], poses[:, :3, 3])
  assert np.allclose(ends[0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-9)
  root = np.sqrt(0.5)
  assert np.allclose(ends[1, 3:7], [root, 0.0, 0.0, root], atol=1e-9)
  middle = Take.poses(standing, track, np.array([0.5]))[0]
  assert np.allclose(middle[:3], [0.5, 0.0, 0.0])
  eighth = [np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]
  assert np.allclose(middle[3:7], eighth, atol=1e-9), (
    "halfway between no turn and a quarter turn is an eighth of a turn"
  )
  assert middle[7] == pytest.approx(0.1)
