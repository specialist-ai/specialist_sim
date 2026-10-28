"""Replay a recorded take and film each simulated view over its own footage.

  uv run --extra dataset python scripts/replay.py \\
    --dataset ~/corpus --session canon002 --output overlay.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from specialist_sim.dataset.take import Take
from specialist_sim.gripper import SLOTS
from specialist_sim.pickplace import Cfg, PickPlaceEnv
from specialist_sim.pickplace.task import Fixed


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--session", required=True)
  parser.add_argument("--layer", help="which state configuration to read")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  cfg = Cfg()
  take = Take(args.dataset, args.session, args.layer, 1.0 / cfg.clock.step_dt)
  size = take.observed(0, 0).shape[0]
  env = PickPlaceEnv(
    Cfg(
      image_size=size,
      cameras={
        slot: take.camera(track, size)
        for slot, track in zip(SLOTS, take.tracks, strict=True)
      },
    )
  )
  env.reset([0], initial_condition=Fixed(take.pose(0), np.array([*env.room.tape, 0.0])))
  writer = imageio_ffmpeg.write_frames(
    str(args.output),
    (size * len(SLOTS), size * 2),
    fps=round(1.0 / take.period),
    macro_block_size=1,
    output_params=["-crf", "20"],
  )
  writer.send(None)
  gaps = []
  for tick in range(len(take.times)):
    observation, _ = env.step(
      take.action_chunk(tick, env.rig.cfg.clock.n_action_steps)[None]
    )
    top = np.concatenate(list(observation.image[0]), axis=1)
    bottom = np.concatenate(
      [
        over(
          take.observed(hand, tick),
          observation.image[0, hand],
          observation.body[0, hand],
        )
        for hand in range(len(SLOTS))
      ],
      axis=1,
    )
    writer.send(np.ascontiguousarray(np.concatenate([top, bottom], axis=0)))
    apart = observation.state[0, -1, :, :3] - take.pose(tick)[:, :3]
    gaps.append(np.linalg.norm(apart, axis=1))
  writer.close()
  error = np.asarray(gaps) * 1000.0
  print(f"wrote {args.output} ({len(take.times)} frames, {size} px)")
  print(
    f"tracking against the solve: {np.median(error):.3f} mm median, "
    f"{np.percentile(error, 95):.3f} mm p95"
  )


def over(real: np.ndarray, image: np.ndarray, body: np.ndarray) -> np.ndarray:
  """The simulated frame over the observed one, with every body outlined."""
  drawn = body >= 0
  out = real.copy()
  out[drawn] = (0.5 * real[drawn] + 0.5 * image[drawn]).astype(np.uint8)
  edge = np.zeros(body.shape, bool)
  edge[:-1] |= body[:-1] != body[1:]
  edge[:, :-1] |= body[:, :-1] != body[:, 1:]
  out[edge] = (0, 255, 0)
  return out


if __name__ == "__main__":
  main()
