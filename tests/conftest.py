import pytest

from gantry import Gantry
from specialist_sim.embodiment import Embodiment, Floating


@pytest.fixture(params=[Floating, Gantry], ids=["floating", "gantry"])
def embodiment(request: pytest.FixtureRequest) -> Embodiment:
  """Each carrier a seam test runs on: the free rig, and a joint chain."""
  carrier: Embodiment = request.param()
  return carrier
