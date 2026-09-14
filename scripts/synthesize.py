"""Demonstrate the task in simulation and write the result as a state layer.

  uv run --extra dataset python scripts/synthesize.py \\
    --dataset ~/corpus_sim --episodes 20 --revision $(git rev-parse HEAD)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from specialist_corpus import schema

from specialist_sim.camera import NOMINAL
from specialist_sim.dataset.layer import record, revision, scale, write
from specialist_sim.pickplace import Cfg, PickPlace
from specialist_sim.pickplace.script import expert
from specialist_sim.pickplace.task import Placed


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--size", type=int, default=Cfg.image_size)
  parser.add_argument("--max-steps", type=int, default=2000)
  parser.add_argument("--revision", type=str, default=None)
  args = parser.parse_args()

  sha = args.revision or revision()
  env = PickPlace(Cfg(image_size=args.size))
  layer = schema.state_layer(scale(args.size), 1.0 / env.cfg.clock.row_dt)
  written, refused = 0, 0
  seed, last = args.seed, args.seed + 2 * args.episodes
  while written < args.episodes and seed < last:
    demonstrator = expert(env, np.random.default_rng(seed))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    episode, report = record(env, demonstrator, seed, args.max_steps)
    if not (demonstrator.done and Placed().reached(report)[0]):
      refused += 1
      print(f"  seed {seed}: the demonstration did not finish the task, dropped")
      seed += 1
      continue
    session = f"{stamp}_sim{seed:04d}"
    path = schema.layer_rrd(args.dataset, layer, session)
    write(path, session, episode, args.size, seed, sha, NOMINAL.hz)
    written += 1
    seed += 1
    print(
      f"  {session}: {sum(len(taken) for taken in episode.frames.values())} images, "
      f"{len(episode.row_times)} "
      f"states, {path.stat().st_size / 1e6:.1f} MB"
    )
  print(f"wrote {written} sessions to {args.dataset / layer}, dropped {refused}")


if __name__ == "__main__":
  main()
