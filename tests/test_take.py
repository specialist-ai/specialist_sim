"""Reading a recorded take back as states the environment can be commanded with."""

import numpy as np
import pytest

pytest.importorskip("specialist_dataset", reason="the dataset extra is not installed")

from specialist_sim.take import Track, _world_from_room, states  # noqa: E402


def test_the_room_frame_stands_midway_between_the_anchors() -> None:
  """The origin is the anchors' midpoint, +y runs from the right anchor to the
  left one, +z is up."""
  left = np.eye(4)
  right = np.eye(4)
  left[:3, 3] = [1.0, 0.5, 0.3]
  right[:3, 3] = [1.0, -0.5, 0.3]
  world_from_room = _world_from_room(left, right)
  assert np.allclose(world_from_room[:3], [1.0, 0.0, 0.3])
  assert np.allclose(abs(world_from_room[3:]), [1.0, 0.0, 0.0, 0.0])


def test_a_pose_is_read_between_the_samples_it_falls_between() -> None:
  times = np.array([0.0, 1.0])
  root = np.sqrt(0.5)
  track = Track(
    slot="cam01",
    tag_from_lens=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    times=times,
    poses=np.array([[0.0, 0, 0, 1, 0, 0, 0], [1.0, 0, 0, root, 0, 0, root]]),
    opening=np.array([0.0, 0.2]),
    image_times=times,
    images=[b"", b""],
  )
  standing = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  ends = states(track, standing, np.array([0.0, 1.0]))
  assert np.allclose(ends[:, :7], track.poses, atol=1e-9)
  middle = states(track, standing, np.array([0.5]))[0]
  assert np.allclose(middle[:3], [0.5, 0.0, 0.0])
  eighth = [np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]
  assert np.allclose(middle[3:7], eighth, atol=1e-9), (
    "halfway between no turn and a quarter turn is an eighth of a turn"
  )
  assert middle[7] == pytest.approx(0.1)
