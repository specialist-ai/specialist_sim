"""Replay a recorded take and film each simulated view over its own footage.

  uv run --extra dataset sim-replay \\
    --dataset ~/dataset_0903 --session 20260903_030121_IAQEB2601UVVAJ \\
    --layer state_s12_r100 --out overlay.mp4
"""

import argparse
from pathlib import Path

import numpy as np

from specialist_sim import video
from specialist_sim.arrays import Int32Array, UInt8Array
from specialist_sim.env import Cfg, Clock, Env, Start, Task
from specialist_sim.gripper import HANDS
from specialist_sim.take import Take


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--session", required=True)
  parser.add_argument("--layer", help="which state configuration to read")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()

  take = Take(args.dataset, args.session, Clock(), args.layer)
  size = take.observed(0, 0).shape[0]
  # An empty room: the overlay is of the hands, whatever task a take recorded.
  lenses = {
    hand: take.lens(track, size) for hand, track in zip(HANDS, take.tracks, strict=True)
  }
  env = Env(Task(), Cfg(lenses=lenses, image_size=size, camera_latency=0.0))
  env.reset([0], start=Start(take.state(0), np.zeros((0, 7)), np.zeros(len(HANDS))))
  gaps = []
  with video.writer(args.out, size * len(HANDS), 2 * size, 1.0 / take.period) as write:
    for frame in range(len(take.image_times)):
      observation, _ = env.step(take.action_chunk(frame)[None])
      seen = [
        over(
          take.observed(index, frame),
          observation.image[0, index],
          observation.body[0, index],
        )
        for index in range(len(HANDS))
      ]
      write(np.concatenate([np.hstack(observation.image[0]), np.hstack(seen)]))
      apart = observation.state[0, -1, :, :3] - take.state(frame)[:, :3]
      gaps.append(np.linalg.norm(apart, axis=1))
  error = np.asarray(gaps) * 1000.0
  print(f"wrote {args.out} ({len(take.image_times)} frames, {size} px)")
  print(
    f"tracking against the solve: {np.median(error):.3f} mm median, "
    f"{np.percentile(error, 95):.3f} mm p95"
  )


def over(real: UInt8Array, image: UInt8Array, body: Int32Array) -> UInt8Array:
  """The simulated frame over the observed one, with every body outlined."""
  drawn = body >= 0
  out = real.copy()
  out[drawn] = (0.5 * real[drawn] + 0.5 * image[drawn]).astype(np.uint8)
  edge = np.zeros(body.shape, bool)
  edge[:-1] |= body[:-1] != body[1:]
  edge[:, :-1] |= body[:, :-1] != body[:, 1:]
  out[edge] = (0, 255, 0)
  return out
