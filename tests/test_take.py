"""Reading a recorded take back as poses the environment can be commanded with."""

from dataclasses import fields
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("specialist_corpus", reason="the corpus extra is not installed")

from specialist_sim.corpus.take import Take, Track  # noqa: E402


class _Standing:
  """A take whose room frame is the world frame."""

  room_from_world = np.eye(4)


def _track(times: np.ndarray, spots: np.ndarray, quats: np.ndarray) -> Track:
  wanted: dict[str, Any] = {
    "hand": "right",
    "slot": "cam01",
    "tag_from_lens": np.eye(4),
    "times": times,
    "translations": spots,
    "quaternions": quats,
    "speeds": np.zeros((len(times), 3)),
    "opening": (times, np.zeros(len(times))),
    "ticks": times,
    "frames": [b"" for _ in times],
  }
  held = {field.name for field in fields(Track)}
  return Track(**{name: value for name, value in wanted.items() if name in held})


def test_a_pose_is_read_between_the_samples_it_falls_between() -> None:
  times = np.array([0.0, 1.0])
  spots = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
  quarter = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
  quats = np.stack([np.array([1.0, 0.0, 0.0, 0.0]), quarter])
  track = _track(times, spots, quats)
  standing = cast(Take, _Standing())
  ends = Take._poses(standing, track, np.array([0.0, 1.0]))
  assert np.allclose(ends[0, :3], spots[0]) and np.allclose(ends[1, :3], spots[1])
  assert np.allclose(ends[0, 3:7], quats[0], atol=1e-9)
  assert np.allclose(ends[1, 3:7], quats[1], atol=1e-9)
  middle = Take._poses(standing, track, np.array([0.5]))[0]
  assert np.allclose(middle[:3], [0.5, 0.0, 0.0])
  eighth = np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)])
  assert np.allclose(middle[3:7], eighth, atol=1e-9), (
    "halfway between no turn and a quarter turn is an eighth of a turn"
  )
