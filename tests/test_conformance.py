"""The environment follows the Gymnasium interface without importing it."""

import subprocess
import sys

import numpy as np
import pytest

from specialist_sim import PickPlaceEnv

gym = pytest.importorskip("gymnasium")
check_env = pytest.importorskip("gymnasium.utils.env_checker").check_env


class _Adapter(gym.Env):  # type: ignore[misc,name-defined]
  metadata = PickPlaceEnv.metadata

  def __init__(self) -> None:
    self._env = PickPlaceEnv()
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space

  def reset(self, *, seed=None, options=None):  # type: ignore[no-untyped-def]
    super().reset(seed=seed)
    return self._env.reset(seed=seed, options=options)

  def step(self, action):  # type: ignore[no-untyped-def]
    return self._env.step(np.asarray(action, np.float64))

  def render(self):  # type: ignore[no-untyped-def]
    return self._env.render()


def test_the_interface_satisfies_the_checker() -> None:
  check_env(_Adapter(), skip_render_check=True)


def test_the_runtime_imports_no_gymnasium() -> None:
  probe = "import sys, specialist_sim; print('gymnasium' in sys.modules)"
  found = subprocess.run(
    [sys.executable, "-c", probe], capture_output=True, text=True, check=True
  )
  assert found.stdout.strip() == "False"
