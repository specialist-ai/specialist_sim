"""The environment follows the Gymnasium interface without importing it."""

import subprocess
import sys

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.embodiment import EmbodimentCfg

gym = pytest.importorskip("gymnasium")
check_env = pytest.importorskip("gymnasium.utils.env_checker").check_env


class _Adapter(gym.Env):  # type: ignore[misc,name-defined]
  metadata = PickPlaceEnv.metadata

  def __init__(self, embodiment: EmbodimentCfg) -> None:
    self._env = PickPlaceEnv(PickPlaceCfg(embodiment=embodiment))
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space

  @staticmethod
  def _alone(observation):  # type: ignore[no-untyped-def]
    """The environment answers for every world; a Gymnasium env has one."""
    out = {"step": observation["step"][0]}
    for slot in [key for key in observation if key != "step"]:
      hand = observation[slot]
      out[slot] = {
        "pose": hand["pose"][0],
        "opening": hand["opening"][0],
        "image": hand["image"][0],
        "image_step": hand["image_step"][0],
      }
    return out

  def reset(self, *, seed=None, options=None):  # type: ignore[no-untyped-def]
    super().reset(seed=seed)
    observation, report = self._env.reset(seed=seed, options=options)
    return self._alone(observation), {k: v[0] for k, v in report.items()}

  def step(self, action):  # type: ignore[no-untyped-def]
    observation, reward, done, cut, report = self._env.step(
      np.asarray(action, np.float64)[None]
    )
    return (
      self._alone(observation),
      float(reward[0]),
      bool(done[0]),
      bool(cut[0]),
      {k: v[0] for k, v in report.items()},
    )

  def render(self):  # type: ignore[no-untyped-def]
    return self._env.render()


def test_the_interface_satisfies_the_checker(embodiment: EmbodimentCfg) -> None:
  check_env(_Adapter(embodiment), skip_render_check=True)


def test_the_runtime_imports_no_gymnasium() -> None:
  probe = "import sys, specialist_sim; print('gymnasium' in sys.modules)"
  found = subprocess.run(
    [sys.executable, "-c", probe], capture_output=True, text=True, check=True
  )
  assert found.stdout.strip() == "False"
