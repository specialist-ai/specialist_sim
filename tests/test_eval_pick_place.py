"""Evaluation on the pick-and-place task, and what a report names it ran under."""

from dataclasses import replace
from pathlib import Path

import pytest

from fixtures.scene import Idle
from specialist_sim.env import Cfg, Env
from specialist_sim.eval import eval_policy, revision, summary
from specialist_sim.tasks.pick_place import PickPlace


def test_a_revision_names_the_model_the_cfg_and_the_code(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  cfg = Cfg(image_size=32, image_hands=())
  env = Env(PickPlace(), cfg, num_envs=2)
  outcomes = eval_policy(env, Idle(), seeds=[3, 4, 5], time_limit=0.4)
  ran = [(outcome.seed, outcome.success) for outcome in outcomes]
  assert ran == [(seed, False) for seed in (3, 4, 5)]
  assert [outcome.time for outcome in outcomes] == pytest.approx([0.4] * 3)
  report = summary(outcomes, env, 0.4)
  assert report["tracking_error"] < 0.01 and report["unstable_rate"] == 0.0
  named = revision(env)
  assert report["revision"] == named
  assert len(named) == 12 and named == revision(Env(PickPlace(), cfg))
  assert named != revision(Env(PickPlace(), replace(cfg, image_size=48)))
  read = Path.read_bytes

  def edited(path: Path) -> bytes:
    return read(path) + (b"#" if path.name == "measures.py" else b"")

  monkeypatch.setattr(Path, "read_bytes", edited)
  assert named != revision(Env(PickPlace(), cfg))
