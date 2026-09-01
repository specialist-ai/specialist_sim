"""A standard set of numbers for judging a policy in this rig.

The definitions live here, in the package that owns the environment they are
statements about, so that two people reporting a success rate are reporting the
same quantity. A training repository scoring a checkpoint imports these rather
than writing its own, and nothing here depends on how a policy is built, what
framework trained it, or how many hands the embodiment has.

    from specialist_sim.metrics import Episode, summarise

    scoring = Episode(env)
    observation, info = env.reset(seeds=seeds)
    scoring.begin(seeds=seeds)
    while scoring.running.any():
      action = policy(observation)
      observation, reward, terminated, truncated, info = env.step(action)
      scoring.record(action, observation, reward, terminated, truncated, info)
    print(summarise(scoring.reports()))

:mod:`specialist_sim.metrics.episode` defines what one episode did and
:mod:`specialist_sim.metrics.summary` how a set of them is reduced; the module
docstrings of each are the specification.
:mod:`specialist_sim.metrics.confidence` is the intervals every summarised
number carries, and :mod:`specialist_sim.metrics.smoothness` the one measure that
is not an outcome. Either can be deleted without touching the rest.
"""

from specialist_sim.metrics.episode import STAGES, Episode, EpisodeReport, MetricsCfg
from specialist_sim.metrics.summary import summarise

__all__ = ["STAGES", "Episode", "EpisodeReport", "MetricsCfg", "summarise"]
