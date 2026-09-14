import pytest

from fixtures.gantry import Gantry
from specialist_sim.embodiment import Embodiment, Floating


@pytest.fixture(params=[Floating, Gantry], ids=["floating", "gantry"])
def embodiment(request: pytest.FixtureRequest) -> type[Embodiment]:
  """Each embodiment a seam test runs on: the free rig, and a joint chain."""
  embodiment: type[Embodiment] = request.param
  return embodiment
