"""Demonstrate the task in simulation and write the result as a state layer.

The cube starts on the tape; one hand kicks it somewhere it was not and the
other picks it up and puts it back, over and over. Each episode becomes a
session a trainer reads exactly as it reads a recorded one; an episode that
does not run its cycles out and end with the cube on the tape is not written.

Episodes are generated wherever the GPU is, which may be a copy of the tree
holding no history, so the revision is passed in from where the history is:

  uv run --extra dataset python scripts/synthesise.py \
    --dataset ~/corpus_sim --episodes 20 --revision $(git rev-parse HEAD)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from specialist_corpus import schema

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.dataset.state import (
  IMAGE_STRIDE,
  layer_scale,
  revision,
  rollout,
  write_layer,
)
from specialist_sim.expert import Expert, ExpertCfg
from specialist_sim.spaces import GRIPPERS


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=10)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--size", type=int, default=240)
  parser.add_argument("--max-steps", type=int, default=2000)
  parser.add_argument("--revision", type=str, default=None)
  parser.add_argument(
    "--cfg",
    action="append",
    default=[],
    metavar="FIELD=VALUE",
    help="Override one ExpertCfg field, e.g. --cfg bite=0.02 --cfg kick=0.35,0.10. "
    "Tuple fields take comma-separated values. Repeatable; a demonstration's "
    "style is what decides whether a dataset matches the recordings, so it is "
    "worth sweeping without editing the defaults.",
  )
  args = parser.parse_args()

  cfg = ExpertCfg()
  for setting in args.cfg:
    field, _, written = setting.partition("=")
    if not hasattr(cfg, field):
      raise SystemExit(f"ExpertCfg has no field {field!r}")
    current = getattr(cfg, field)
    parsed: object
    if isinstance(current, tuple):
      parsed = tuple(float(part) for part in written.split(","))
      if len(parsed) != len(current):
        raise SystemExit(f"{field} takes {len(current)} values, not {len(parsed)}")
    else:
      parsed = float(written)
    cfg = replace(cfg, **{field: parsed})
  if args.cfg:
    print("expert overrides: " + "  ".join(args.cfg))

  sha = args.revision or revision()
  env = PickPlaceEnv(PickPlaceCfg(image_size=args.size, max_steps=10**6))
  scale = layer_scale(env.cameras[GRIPPERS[0]].width, args.size)
  layer = schema.state_layer(scale, IMAGE_STRIDE)
  written, refused = 0, 0
  # A draw that does not finish the task is not a session, so drawing goes on
  # until the count asked for is the count written.
  seed, last = args.seed, args.seed + 2 * args.episodes
  while written < args.episodes and seed < last:
    expert = Expert(env, np.random.default_rng(seed), cfg)
    started_at = time.time_ns()
    kept = rollout(env, expert, seed, args.max_steps)
    # An episode runs many cycles, so the one that matters is the last, and a
    # rollout cut short by its step limit stops in the middle of one.
    if not (expert.done and env.judge.info()["settled"]):
      refused += 1
      why = (
        "the cube left the table"
        if env.judge.info()["dropped"]
        else "it ran out of steps mid-cycle"
        if not expert.done
        else "it stopped with the cube off the tape"
      )
      print(f"  seed {seed}: the demonstration did not finish the task -- {why}")
      seed += 1
      continue
    session = f"sim{seed:04d}"
    path = schema.layer_rrd(args.dataset, layer, session)
    write_layer(path, session, kept, args.size, seed, scale, started_at, sha)
    written += 1
    seed += 1
    print(
      f"  {session}: {len(kept.frame_times)} images, {len(kept.step_times)} states, "
      f"{path.stat().st_size / 1e6:.1f} MB"
    )
  print(f"wrote {written} sessions to {args.dataset / layer}, dropped {refused}")


if __name__ == "__main__":
  main()
