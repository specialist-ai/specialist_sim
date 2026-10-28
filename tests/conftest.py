import pytest

from gantry import Gantry
from specialist_sim.embodiment import Embodiment, Floating


@pytest.fixture(params=[Floating, Gantry], ids=["floating", "gantry"])
def embodiment(request: pytest.FixtureRequest) -> Embodiment:
  """Each embodiment a seam test runs on: the free rig, and a joint chain."""
  embodiment: Embodiment = request.param()
  return embodiment
