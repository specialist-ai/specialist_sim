"""Replay a recorded take through the environment and film it beside what the
policy saw.

A step tracks a trajectory forward by one control step, so each tick is driven
by the trajectory that lands on it. The observation layer sets the tick rate,
so the environment runs at the rate the policy is trained at.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.corpus.take import Take
from specialist_sim.observation import flat_row
from specialist_sim.render.renderer import View
from specialist_sim.spaces import GRIPPERS


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=Path, required=True)
  parser.add_argument("--session", required=True)
  parser.add_argument("--layer", help="which observation configuration to read")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  take = Take(args.dataset, args.session, args.layer)
  size = take.observed(0, 0).shape[0]
  ticks = len(take.times)
  # A 29.97 fps capture puts its keyframes 100.1 ms apart, so the environment
  # steps at the nominal rate and the trajectory is sampled at the real instants.
  rate = round(1.0 / take.period)

  env = PickPlaceEnv(
    PickPlaceCfg(
      control_hz=rate,
      max_steps=ticks + 2,
      image_size=size,
      cameras={
        slot: take.camera(track, size)
        for slot, track in zip(GRIPPERS, take.tracks, strict=True)
      },
    )
  )
  env.reset(seed=0)

  writer = imageio_ffmpeg.write_frames(
    str(args.output),
    (size * len(GRIPPERS), size * 2),
    fps=rate,
    macro_block_size=1,
    output_params=["-crf", "20"],
  )
  writer.send(None)
  gaps = []
  for tick in range(ticks):
    observation, _, _, _, _ = env.step(take.trajectory(tick))
    top = [observation[slot]["image"][0] for slot in GRIPPERS]
    bottom = [
      _over(take.observed(hand, tick), env.views[slot])
      for hand, slot in enumerate(GRIPPERS)
    ]
    writer.send(
      np.ascontiguousarray(
        np.concatenate(
          [np.concatenate(top, axis=1), np.concatenate(bottom, axis=1)], axis=0
        )
      )
    )
    wanted = take.pose(tick)
    apart = (flat_row(observation) - wanted).reshape(len(GRIPPERS), -1)[:, :3]
    gaps.append(np.linalg.norm(apart, axis=1))
  writer.close()
  error = np.asarray(gaps) * 1000.0
  print(f"wrote {args.output} ({ticks} frames at {rate:.2f} Hz, {size} px)")
  print(
    f"tracking against the solve: {np.median(error):.3f} mm median, "
    f"{np.percentile(error, 95):.3f} mm p95"
  )


def _over(real: np.ndarray, view: View) -> np.ndarray:
  """The simulated frame over the observed one, with every body outlined."""
  drawn = view.body >= 0
  out = real.copy()
  out[drawn] = (0.5 * real[drawn] + 0.5 * view.image[drawn]).astype(np.uint8)
  out[_borders(view.body)] = (0, 255, 0)
  return out


def _borders(body: np.ndarray) -> np.ndarray:
  """Where one body gives way to another, one pixel wide."""
  edge = np.zeros(body.shape, bool)
  edge[:-1] |= body[:-1] != body[1:]
  edge[:, :-1] |= body[:, :-1] != body[:, 1:]
  return edge


if __name__ == "__main__":
  main()
