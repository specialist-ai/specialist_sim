"""The standard metrics: what each number means, and that it means it.

These are the definitions other repositories are meant to import instead of
writing their own, so what is checked here is the definitions themselves --
especially the ones where a lazier definition would look identical most of the
time and disagree exactly where it matters.
"""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.action import hold
from specialist_sim.metrics import STAGES, Episode, MetricsCfg, summarise
from specialist_sim.metrics.confidence import wilson
from specialist_sim.observation import flat_row, flat_rows
from specialist_sim.randomise import place_cube
from specialist_sim.spaces import ACTION_HIGH, STATE_LOW
from specialist_sim.task import TaskCfg


def _held(env: PickPlaceEnv, observation: dict[str, Any]) -> np.ndarray:
  """The action that asks every world to stay exactly where it is."""
  return np.stack([hold(row) for row in flat_rows(observation)])


def _run(
  env: PickPlaceEnv, scoring: Episode, steps: int, action: Any = None
) -> dict[str, Any]:
  observation, _ = env.reset(seed=0)
  scoring.begin(seeds=[0] * env.worlds)
  for _ in range(steps):
    commanded = _held(env, observation) if action is None else action
    observation, reward, terminated, truncated, info = env.step(commanded)
    scoring.record(commanded, observation, reward, terminated, truncated, info)
    if not scoring.running.any():
      break
  return {"observation": observation}


def test_an_episode_that_does_nothing_truncates_and_says_so() -> None:
  env = PickPlaceEnv(PickPlaceCfg(max_steps=8, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 20)
  report = scoring.report()
  assert report.ended == "truncated"
  assert report.steps == 8
  assert not report.success and not report.picked and not report.dropped
  assert report.steps_to_success == -1
  assert report.first_contact == -1
  assert report.reward == 0.0


def test_a_success_the_hand_never_carried_is_not_a_pick() -> None:
  """The distinction ``picked`` exists for.

  The cube is taken off the tape and put back by hand, so the task latches
  without a pad ever touching it and without it leaving the table. A success
  rate alone would call that a solved episode; ``picked`` is what says the hand
  had nothing to do with it.
  """
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400, image_size=64))
  scoring = Episode(env)
  observation, _ = env.reset(seed=0, options={"cube": (0.60, -0.10)})
  scoring.begin(seeds=[0])
  action = hold(flat_row(observation).astype(np.float64))
  observation, reward, terminated, truncated, info = env.step(action)
  scoring.record(action, observation, reward, terminated, truncated, info)
  place_cube(env.scene, env.scene.data, env.scene.tape)
  for _ in range(TaskCfg().settle_steps + 3):
    observation, reward, terminated, truncated, info = env.step(action)
    scoring.record(action, observation, reward, terminated, truncated, info)
    if not scoring.running.any():
      break
  report = scoring.report()
  assert report.success, "the task never latched"
  assert report.ended == "success"
  assert not report.picked, "a cube placed by hand was scored as a pick"
  assert report.held_steps == 0
  assert report.lift < MetricsCfg().lift
  assert report.steps_to_success == report.steps
  assert report.reward == 1.0, "an undiscounted return is the success indicator"


def test_a_cube_under_the_table_ends_the_episode_as_dropped() -> None:
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400, image_size=64))
  scoring = Episode(env)
  observation, _ = env.reset(seed=0)
  scoring.begin(seeds=[0])
  address = env.scene.cube.qpos
  env.scene.data.qpos[address + 2] = env.scene.table_top - 1.0
  action = hold(flat_row(observation).astype(np.float64))
  for _ in range(3):
    observation, reward, terminated, truncated, info = env.step(action)
    scoring.record(action, observation, reward, terminated, truncated, info)
    if not scoring.running.any():
      break
  report = scoring.report()
  assert report.dropped and report.ended == "dropped"
  assert not report.success and not report.picked


@pytest.mark.parametrize("steps", [2, 6])
def test_a_command_outside_the_action_box_is_a_fraction_not_a_count(
  steps: int,
) -> None:
  """The share of what was commanded, so it compares across rigs and run lengths.

  One column of the trajectory is past the box, out of `hands * PER_HAND` of
  them, and that share is the answer however many steps are run. A count would
  grow with the episode and would halve on a one-armed rig for identical
  behaviour.
  """
  env = PickPlaceEnv(PickPlaceCfg(max_steps=steps, image_size=64))
  scoring = Episode(env)
  observation, _ = env.reset(seed=0)
  scoring.begin()
  action = hold(flat_row(observation).astype(np.float64))
  action[:, 0] = 1e6
  for _ in range(steps):
    observation, reward, terminated, truncated, info = env.step(action)
    scoring.record(action, observation, reward, terminated, truncated, info)
  assert scoring.report().clipped == pytest.approx(1.0 / ACTION_HIGH.shape[1])


def test_tracking_is_zero_when_the_hands_are_asked_to_stay_put() -> None:
  env = PickPlaceEnv(PickPlaceCfg(max_steps=6, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 6)
  report = scoring.report()
  assert report.tracking < 1e-4, "a held hand was reported as mistracking"
  assert report.tracking_rotation < 1e-6


def test_a_hand_that_cannot_get_where_it_was_sent_is_reported_as_late() -> None:
  """Tracking is the rig's fidelity, and it is not always zero.

  A hand pulled by a stiff wrench crosses the whole action box inside one
  control step, so the only way to see it behind its setpoint is to stop it --
  here with the tabletop, which the bottom of the box sits level with.
  """
  env = PickPlaceEnv(PickPlaceCfg(max_steps=6, image_size=64))
  scoring = Episode(env)
  observation, _ = env.reset(seed=0)
  scoring.begin()
  commanded = flat_row(observation).astype(np.float64)
  commanded[2] = STATE_LOW[2]  # straight down, into the tabletop
  action = hold(commanded)
  for _ in range(6):
    observation, reward, terminated, truncated, info = env.step(action)
    scoring.record(action, observation, reward, terminated, truncated, info)
  report = scoring.report()
  assert report.tracking > 0.01, "a blocked hand was reported as on target"
  assert report.tracking_rotation > 0.01, "the wrist was not held off either"


def test_a_stale_frame_is_reported_at_its_age() -> None:
  lag = 7
  env = PickPlaceEnv(PickPlaceCfg(max_steps=5, camera_lag=lag, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 5)
  assert scoring.report().image_age == lag


def test_each_world_is_scored_on_its_own_episode() -> None:
  """One world ending early leaves its report alone while the rest run on."""
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400, image_size=64), worlds=2)
  scoring = Episode(env)
  observation, _ = env.reset(seeds=[0, 1])
  scoring.begin(seeds=[0, 1])
  address = env.scene.cube.qpos
  env.datas[0].qpos[address + 2] = env.scene.table_top - 1.0
  action = _held(env, observation)
  for _ in range(6):
    observation, reward, terminated, truncated, info = env.step(action)
    scoring.record(action, observation, reward, terminated, truncated, info)
  first, second = scoring.reports()
  assert first.dropped and first.ended == "dropped"
  assert not second.dropped and second.ended == "running"
  assert first.steps < second.steps, "the ended world kept accumulating"
  assert (first.seed, second.seed) == (0, 1)


def test_a_summary_is_a_rate_over_all_and_a_mean_over_the_ones_it_applies_to() -> None:
  env = PickPlaceEnv(PickPlaceCfg(max_steps=3, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 3)
  losing = replace(scoring.report(), seed=0)
  winning = replace(losing, seed=1, success=True, steps_to_success=2)
  summary = summarise([losing, winning])
  assert summary["episodes"] == 2
  assert summary["success_rate"] == 0.5
  assert summary["steps_to_success"] == 2.0, "a loss was averaged into the wins"
  assert summary["steps_to_success_n"] == 1, "the count it was taken over is missing"
  assert summary["steps"] == 3.0
  assert np.isnan(summarise([losing])["steps_to_success"]), "no wins is not zero"
  assert np.isnan(summarise([])["success_rate"]), "no episodes is not zero"
  assert summarise([])["episodes"] == 0


def test_every_reported_number_carries_an_interval() -> None:
  """A point estimate on its own is the commonest way an evaluation overstates.

  The interval has to bracket the estimate, and it has to be wide at these
  sample sizes: two episodes say almost nothing, and the summary should say so
  rather than quote 50% as though it were known.
  """
  env = PickPlaceEnv(PickPlaceCfg(max_steps=3, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 3)
  losing = replace(scoring.report(), seed=0)
  winning = replace(losing, seed=1, success=True, steps_to_success=2)
  summary = summarise([losing, winning])
  for name in ("success_rate", "pick_rate", "lift", "tracking", "progress"):
    low, high = summary[f"{name}_low"], summary[f"{name}_high"]
    assert low <= summary[name] <= high, name
  assert summary["success_rate_high"] - summary["success_rate_low"] > 0.5, (
    "two episodes were reported as though they pinned a rate down"
  )


def test_a_summary_refuses_to_count_one_episode_twice() -> None:
  """Two reports on one seed are not two episodes, and every rate would be wrong."""
  env = PickPlaceEnv(PickPlaceCfg(max_steps=3, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 3)
  one = replace(scoring.report(), seed=4)
  with pytest.raises(ValueError, match="repeat a seed"):
    summarise([one, one])


def test_a_summary_refuses_an_episode_that_has_not_ended() -> None:
  """Half an episode's numbers are not an episode's numbers."""
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 2)
  with pytest.raises(ValueError, match="have not ended"):
    summarise(scoring.reports())


def _reach_toward(
  env: PickPlaceEnv, scoring: Episode, share: float, steps: int
) -> None:
  """Close `share` of the gap to the cube each step, with the left hand."""
  observation, info = env.reset(seed=0)
  scoring.begin(seeds=[0])
  cube = np.asarray(info["cube_pos"][0], np.float64)
  for _ in range(steps):
    row = flat_row(observation).astype(np.float64)
    row[:3] += share * (cube - row[:3])
    action = hold(row)
    observation, reward, terminated, truncated, info = env.step(action)
    scoring.record(action, observation, reward, terminated, truncated, info)


def test_partial_credit_separates_two_failures() -> None:
  """The reason the stage metric exists.

  A policy frozen in place and a policy that drove a hand onto the cube but
  never closed on it are both `success=False`, and a binary metric calls them
  equal. Grading them apart is what lets an evaluation tell two failing policies
  apart at all, and it is what makes a comparison need far fewer episodes.
  """
  idle = Episode(PickPlaceEnv(PickPlaceCfg(max_steps=20, image_size=64)))
  _run(idle.env, idle, 20)
  reaching = Episode(PickPlaceEnv(PickPlaceCfg(max_steps=20, image_size=64)))
  _reach_toward(reaching.env, reaching, 0.25, 20)

  frozen, tried = idle.report(), reaching.report()
  assert not frozen.success and not tried.success, "neither episode won"
  assert frozen.progress < tried.progress, "two failures were scored the same"
  assert frozen.stage == 0 and tried.stage >= 1
  assert tried.closest_approach < frozen.closest_approach


def test_progress_is_the_furthest_stage_reached_and_agrees_with_success() -> None:
  """A completed episode is complete however it got there.

  The cube is put back by hand, so no pad ever touches it and the intermediate
  flags never fire — but the task did latch, and progress says so. `picked` is
  the metric that records the hand had nothing to do with it, and the two
  together are the whole story where either alone would mislead.
  """
  env = PickPlaceEnv(PickPlaceCfg(max_steps=400, image_size=64))
  scoring = Episode(env)
  observation, _ = env.reset(seed=0, options={"cube": (0.60, -0.10)})
  scoring.begin(seeds=[0])
  action = hold(flat_row(observation).astype(np.float64))
  observation, reward, terminated, truncated, info = env.step(action)
  scoring.record(action, observation, reward, terminated, truncated, info)
  place_cube(env.scene, env.scene.data, env.scene.tape)
  for _ in range(TaskCfg().settle_steps + 3):
    observation, reward, terminated, truncated, info = env.step(action)
    scoring.record(action, observation, reward, terminated, truncated, info)
    if not scoring.running.any():
      break
  report = scoring.report()
  assert report.success
  assert report.stage == len(STAGES) and report.progress == 1.0
  assert not report.picked, "progress must not be read as the hand having done it"


def test_smoothness_tells_a_gliding_hand_from_a_juddering_one() -> None:
  """Two policies can succeed alike and move nothing alike.

  Both hands here cover the same ground; one is asked for it steadily and the
  other in stop-start lurches. Nothing in the outcome metrics separates them.
  """
  travel, steps = 0.08, 16

  def drive(lurch: bool) -> float:
    env = PickPlaceEnv(PickPlaceCfg(max_steps=steps + 2, image_size=64))
    scoring = Episode(env)
    observation, _ = env.reset(seed=0)
    scoring.begin(seeds=[0])
    start = flat_row(observation).astype(np.float64)
    for step in range(steps):
      row = start.copy()
      share = (step + 1) / steps
      row[1] += travel * (round(share * 4) / 4 if lurch else share)
      action = hold(row)
      observation, reward, terminated, truncated, info = env.step(action)
      scoring.record(action, observation, reward, terminated, truncated, info)
    return scoring.report().smoothness

  gliding, juddering = drive(lurch=False), drive(lurch=True)
  assert np.isfinite(gliding) and np.isfinite(juddering)
  assert gliding < 0.0 and juddering < 0.0, "a spectral arc length is negative"
  assert juddering < gliding, "the juddering hand was not scored as less smooth"


def test_a_stage_rate_counts_what_happened_not_what_it_implies() -> None:
  """The stages are not strictly ordered, and pretending they are invents grasps.

  A cube can be knocked into the air by the back of a gripper, which is `lifted`
  with no pad ever on it. Reading a touch out of a lift would report a grasp that
  did not occur, so each rate counts its own event and `progress` carries the
  monotone reading instead.
  """
  env = PickPlaceEnv(PickPlaceCfg(max_steps=3, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 3)
  knocked = replace(scoring.report(), seed=0, stages=("lifted",), stage=3, progress=0.6)
  summary = summarise([knocked])
  assert summary["lifted_rate"] == 1.0
  assert summary["touched_rate"] == 0.0, "a grasp was invented from a lift"
  assert summary["progress"] == 0.6, "the monotone reading is still there"
  assert summary["placed_rate"] == summary["success_rate"]


def test_the_stage_rates_read_as_a_failure_breakdown() -> None:
  idle = Episode(PickPlaceEnv(PickPlaceCfg(max_steps=20, image_size=64)))
  _run(idle.env, idle, 20)
  reaching = Episode(PickPlaceEnv(PickPlaceCfg(max_steps=20, image_size=64)))
  _reach_toward(reaching.env, reaching, 0.25, 20)
  summary = summarise(
    [replace(idle.report(), seed=0), replace(reaching.report(), seed=1)]
  )
  assert summary["reached_rate"] == 0.5, "one of the two episodes reached the cube"
  assert summary["placed_rate"] == summary["success_rate"] == 0.0


@pytest.mark.parametrize(
  ("wins", "trials"),
  [(0, 1), (0, 20), (1, 20), (10, 20), (19, 20), (20, 20), (1, 1)],
)
def test_a_wilson_interval_stays_inside_the_range_it_describes(
  wins: int, trials: int
) -> None:
  """Including at the ends, where a rate is 0 or 1 and rounding wants to escape."""
  low, high = wilson(wins, trials)
  assert 0.0 <= low <= wins / trials <= high <= 1.0
  assert low == 0.0 if wins == 0 else low > 0.0
  assert high == 1.0 if wins == trials else high < 1.0


def test_a_wilson_interval_narrows_as_episodes_are_added() -> None:
  """The whole point: more episodes, less doubt, at the same observed rate."""
  widths = [
    wilson(round(0.6 * trials), trials)[1] - wilson(round(0.6 * trials), trials)[0]
    for trials in (10, 20, 50, 200, 1000)
  ]
  assert widths == sorted(widths, reverse=True)
  assert widths[1] > 0.35, "twenty episodes were reported as pinning a rate down"
  assert widths[-1] < 0.08


def test_an_episode_the_caller_ends_is_finished_data() -> None:
  """A step budget is the caller's rule, and the environment knows nothing of it.

  Without a way to say so the episode stays `running` and the whole set is
  refused, which is what happens to any evaluation that bounds its own driving
  rather than the environment's horizon.
  """
  env = PickPlaceEnv(PickPlaceCfg(max_steps=10**6, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 3)
  assert scoring.report().ended == "running"
  with pytest.raises(ValueError, match="have not ended"):
    summarise(scoring.reports())

  scoring.stop()
  report = scoring.report()
  assert report.ended == "stopped" and report.steps == 3
  assert not scoring.running.any()
  summary = summarise([replace(report, seed=0)])
  assert summary["stopped_rate"] == 1.0
  assert summary["truncated_rate"] == 0.0, "the caller's budget is not the horizon"


def test_the_caller_may_not_claim_an_outcome_that_is_the_environments() -> None:
  env = PickPlaceEnv(PickPlaceCfg(max_steps=10**6, image_size=64))
  scoring = Episode(env)
  _run(env, scoring, 2)
  for taken in ("success", "dropped", "truncated"):
    with pytest.raises(ValueError, match="environment's to say"):
      scoring.stop(reason=taken)
