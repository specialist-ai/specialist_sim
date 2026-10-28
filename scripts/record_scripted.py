"""Record LEGO sorting episodes with the scripted policy, headless, as fast as
the physics runs: the fallback to teleoperation with ``scripts/record.py``.

  uv run python scripts/record_scripted.py --out data/episodes/scripted_drop \\
    --episodes 100 --hands right_gripper --bricks 1

Each episode is a fresh tray, ``--seed`` plus its index, written as
``<mode>_<seed>.npz``. Trays the script did not clear are kept unless
``--success-only``: a failure is data about the scene too.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from specialist_sim import episode
from specialist_sim.gripper import SLOTS
from specialist_sim.sort_lego import Cfg, SortLegoEnv
from specialist_sim.sort_lego.initial import MODES
from specialist_sim.sort_lego.script import scripted_policy


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--out", type=Path, required=True, help="where the episodes are written"
  )
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument(
    "--mode", choices=MODES, default="drop", help="how a tray is laid out"
  )
  parser.add_argument(
    "--bricks",
    type=int,
    default=None,
    help="bricks on a tray (default: the mode's count)",
  )
  parser.add_argument(
    "--hands", nargs="+", choices=SLOTS, default=list(SLOTS), help="the hands that work"
  )
  parser.add_argument(
    "--seed", type=int, default=0, help="the first episode's layout seed"
  )
  parser.add_argument(
    "--timeout", type=float, default=60.0, help="the longest an episode runs, seconds"
  )
  parser.add_argument(
    "--success-only", action="store_true", help="keep only trays the script cleared"
  )
  args = parser.parse_args()

  env = SortLegoEnv(
    Cfg(render=False, driven=tuple(args.hands), mode=args.mode, bricks=args.bricks)
  )
  steps = round(args.timeout / env.cfg.clock.step_dt)
  kept = 0
  for index in range(args.episodes):
    started = time.perf_counter()
    seed = args.seed + index
    observation, report = env.reset([seed])
    policy = scripted_policy(env, 0, np.random.default_rng(seed))
    recorder = episode.Recorder()
    recorder.start(f"{args.mode}_{seed:06d}", {"source": "scripted", **env.describe(0)})
    recorder.extend(*(rows[-1:] for rows in env.rows(0)))
    for _ in range(steps):
      action = policy.predict_action_chunk(observation.state[0, -1], env.bricks(0))
      observation, report = env.step(action[None])
      recorder.extend(*env.rows(0))
      if report.done[0] or policy.done:
        break
    score = env.score(0)
    flown = recorder.finish({"score": score.as_dict()})
    if score.cleared or not args.success_only:
      episode.save(args.out, flown)
      kept += 1
    print(
      f"{flown.name}: {score.sorted}/{score.started} sorted, {flown.timestamp[-1]:.1f} s "
      f"in {time.perf_counter() - started:.1f} s"
    )
  print(f"{kept} episodes in {args.out}")


if __name__ == "__main__":
  main()
