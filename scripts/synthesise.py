"""Demonstrate the task in simulation and write the result as a state layer.

The cube starts on the tape; one hand kicks it somewhere it was not and the
other picks it up and puts it back, over and over. Each episode becomes a
session a trainer reads exactly as it reads a recorded one; an episode that
does not run its cycles out and end with the cube on the tape is not written.

Episodes are generated wherever the GPU is, which may be a copy of the tree
holding no history, so the revision is passed in from where the history is:

  uv run --extra corpus python scripts/synthesise.py \
    --corpus ~/corpus_sim --episodes 20 --revision $(git rev-parse HEAD)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from specialist_corpus import schema

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.corpus.state import (
  IMAGE_STRIDE,
  layer_scale,
  revision,
  rollout,
  write_layer,
)
from specialist_sim.expert import Expert
from specialist_sim.spaces import GRIPPERS


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--corpus", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--size", type=int, default=240)
  parser.add_argument("--max-steps", type=int, default=2000)
  parser.add_argument("--revision", type=str, default=None)
  args = parser.parse_args()

  sha = args.revision or revision()
  env = PickPlaceEnv(PickPlaceCfg(image_size=args.size, max_steps=10**6))
  scale = layer_scale(env.cameras[GRIPPERS[0]].width, args.size)
  layer = schema.state_layer(scale, IMAGE_STRIDE)
  written, refused = 0, 0
  # A draw that does not finish the task is not a session, so drawing goes on
  # until the count asked for is the count written.
  seed, last = args.seed, args.seed + 2 * args.episodes
  while written < args.episodes and seed < last:
    expert = Expert(env, np.random.default_rng(seed))
    started_at = time.time_ns()
    kept = rollout(env, expert, seed, args.max_steps)
    # An episode runs many cycles, so the one that matters is the last, and a
    # rollout cut short by its step limit stops in the middle of one.
    if not (expert.done and env.judge.info()["settled"]):
      refused += 1
      print(f"  seed {seed}: the demonstration did not finish the task, dropped")
      seed += 1
      continue
    session = f"sim{seed:04d}"
    path = schema.layer_rrd(args.corpus, layer, session)
    write_layer(path, session, kept, args.size, seed, scale, started_at, sha)
    written += 1
    seed += 1
    print(
      f"  {session}: {len(kept.frame_times)} images, {len(kept.step_times)} states, "
      f"{path.stat().st_size / 1e6:.1f} MB"
    )
  print(f"wrote {written} sessions to {args.corpus / layer}, dropped {refused}")


if __name__ == "__main__":
  main()
