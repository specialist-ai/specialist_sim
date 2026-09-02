"""The bridge to a corpus: what an evaluation reads off the data a policy was
trained on. Reading a catalog needs the ``corpus`` extra; the rules that turn
its rows into a start region do not."""

from specialist_sim.corpus.region import region

__all__ = ["region"]
