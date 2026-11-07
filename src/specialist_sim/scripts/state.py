"""Write recorded episodes as the state layers specialist's scripts/state.py
writes for a captured session. It needs a GPU.

  uv run --extra dataset sim-state ~/recordings/NAME/*.npz --output-dir ~/dataset_sim
"""

import argparse
import math
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
from rerun.urdf import UrdfTree
from specialist_camera import MASK_RADIUS
from specialist_dataset import schema
from specialist_dataset.layer import encode, hand_chunks, write_layer
from specialist_description import get_urdf_path

from specialist_sim import episode, rotation
from specialist_sim.arrays import Float64Array, Int64Array
from specialist_sim.camera.lens import Lens
from specialist_sim.camera.shutter import CAMERA_HZ
from specialist_sim.env import make
from specialist_sim.eval import revision
from specialist_sim.gripper import HANDS
from specialist_sim.take import SIDE

# Pixels across a capture's decoded rear lens; --image-size divides it.
LENS_WIDTH = 2880
URDF = get_urdf_path("specialist_handheld")


def exposures(
  offset: Float64Array, rows: int, f_row: float
) -> tuple[Float64Array, Int64Array, Int64Array]:
  """The frames lenses at ``offset`` frames expose inside ``rows`` rows: shutter
  times ``(lenses, m)``, shared label rows ``(m,)`` and capture rows ``(lenses, m)``."""
  frame = np.arange(math.ceil(rows / f_row * CAMERA_HZ) + 1)
  times = (frame + offset[:, None]) / CAMERA_HZ
  label = np.rint(frame / CAMERA_HZ * f_row).astype(np.int64)
  capture = np.rint(times * f_row).astype(np.int64)
  kept = ((times >= 0.0) & (capture < rows)).all(axis=0)
  return times[:, kept], label[kept], capture[:, kept]


def twist(state: Float64Array, f_row: float) -> tuple[Float64Array, Float64Array]:
  """Each hand's world-frame velocity and rotation rate at every row by central
  difference, one-sided at the two ends."""
  behind, ahead = np.clip(np.arange(len(state)) + [[-1], [1]], 0, len(state) - 1)
  span = ((ahead - behind) / f_row)[:, None, None]
  turn = rotation.multiply(
    state[ahead, :, 3:7], rotation.conjugate(state[behind, :, 3:7])
  )
  return (state[ahead, :, :3] - state[behind, :, :3]) / span, rotation.log(turn) / span


def write(
  path: Path, dataset: Path, hands: tuple[str, ...], scale: int, supersample: int
) -> Path:
  """The state layer of ``hands`` the episode at ``path`` makes under ``dataset``."""
  recorded = episode.load(path)
  state, qpos, n = recorded.rows.state, recorded.rows.qpos, len(recorded.rows.state)
  described = make(recorded.meta)
  f_row = 1.0 / described.cfg.clock.row_dt
  rng = np.random.default_rng(zlib.crc32(recorded.name.encode()))
  offset = np.concatenate([[0.0], rng.uniform(-0.5, 0.5, len(hands) - 1)])
  shutter, label, capture = exposures(offset, n, f_row)
  lin_vel, ang_vel = twist(state, f_row)
  width = LENS_WIDTH // scale
  chunks = []
  for k, hand in enumerate(hands):
    slot, i = f"cam{k:02d}", HANDS.index(hand)
    rig = replace(described.cfg.lenses.get(hand, Lens()), mask_radius=MASK_RADIUS)
    env = make(
      recorded.meta,
      image_size=width,
      supersample=supersample,
      image_hands=(hand,),
      lenses={**described.cfg.lenses, hand: rig},
    )
    blobs = [encode(env.trace(q[None])[0][0, 0]) for q in qpos[capture[k]]]
    q = state[:, i, [4, 5, 6, 3]]
    chunks += hand_chunks(
      hand,
      slot,
      UrdfTree.from_file_path(str(URDF), frame_prefix=f"{slot}_"),
      (shutter[k], label, capture[k]),
      blobs,
      np.arange(n),
      (
        np.arange(n) / f_row,
        state[:, i, :3],
        np.where(q[:, 3:] < 0.0, -q, q),
        lin_vel[:, i],
        ang_vel[:, i],
      ),
      state[:, i, 7],
      width,
    )
  account = {
    "config": schema.state_layer(scale, f_row),
    "f_row": f_row,
    "scale": scale,
    "mask_radius": MASK_RADIUS,
    "side": SIDE,
    "supersample": supersample,
    "task": recorded.meta["task"],
    "revision": revision(described),
  }
  out = schema.layer_rrd(dataset, schema.state_layer(scale, f_row), recorded.name)
  write_layer(out, recorded.name, account, chunks)
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("episodes", type=Path, nargs="+")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--hands", choices=("left", "right", "both"), default="right")
  parser.add_argument("--image-size", type=int, default=360)
  parser.add_argument("--supersample", type=int, default=8)
  args = parser.parse_args()
  hands = HANDS if args.hands == "both" else (args.hands,)
  scale = LENS_WIDTH // args.image_size
  for path in args.episodes:
    print(write(path, args.output_dir, hands, scale, args.supersample))
