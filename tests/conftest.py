import pytest

from gantry import GantryCfg
from specialist_sim.embodiment import EmbodimentCfg, FloatingCfg


@pytest.fixture(
  scope="module", params=[FloatingCfg(), GantryCfg()], ids=["floating", "gantry"]
)
def embodiment(request: pytest.FixtureRequest) -> EmbodimentCfg:
  """Each carrier a seam test runs on: the free rig, and a joint chain."""
  chosen: EmbodimentCfg = request.param
  return chosen
