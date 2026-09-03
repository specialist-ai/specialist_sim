"""The bridge to a corpus: a session read into the environment, the region a
policy's picks begin in, and an episode written back out as one."""

from specialist_sim.corpus.region import region
from specialist_sim.corpus.state import Episode, rollout, write_layer
from specialist_sim.corpus.take import SIDE, Take, Track

__all__ = ["SIDE", "Episode", "Take", "Track", "region", "rollout", "write_layer"]
