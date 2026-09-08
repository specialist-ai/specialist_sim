"""The bridge to a dataset: a session read into the environment, the region a
policy's picks begin in, and an episode written back out as one."""

from specialist_sim.dataset.region import region
from specialist_sim.dataset.state import Episode, rollout, write_layer
from specialist_sim.dataset.take import SIDE, Take, Track

__all__ = ["SIDE", "Episode", "Take", "Track", "region", "rollout", "write_layer"]
