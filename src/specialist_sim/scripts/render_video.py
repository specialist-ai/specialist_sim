"""Film a recorded episode: an overview of its scene beside each hand's wrist
lens, traced from the recorded qpos in the env the episode names.

  MUJOCO_GL=egl uv run --extra dataset sim-render ~/recordings/NAME/EPISODE.npz --out video.mp4

The overview is MuJoCo's renderer looking at the task's focus from the
operator's side, drawing only the geom group the lenses draw. It needs the
``dataset`` extra, for PyAV, and a GPU.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from specialist_sim import episode, video
from specialist_sim.env import make
from specialist_sim.gripper import DRAWN, HANDS

# Pixels across each tile; a lens is traced at twice that and averaged.
TILE = 360


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("episode", type=Path)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument(
    "--speed", type=float, default=1.0, help="playback speed, 0.5 is half speed"
  )
  args = parser.parse_args()

  recorded = episode.load(args.episode)
  env = make(recorded.meta, image_size=TILE, supersample=2, image_hands=HANDS)
  data = mujoco.MjData(env.model)
  rows = recorded.rows
  overview = mujoco.Renderer(env.model, TILE, TILE)
  shown = mujoco.MjvOption()
  shown.geomgroup[:] = [group == DRAWN for group in range(len(shown.geomgroup))]
  shown.sitegroup[:] = 0
  camera = mujoco.MjvCamera()
  camera.lookat[:] = env.task.focus
  camera.distance = 0.9
  camera.azimuth = 0.0  # from the operator's side, looking toward +x
  camera.elevation = -50.0
  row_hz = 1.0 / env.cfg.clock.row_dt
  # Rows per frame, for a film as near 30 fps as the rows divide.
  stride = max(1, round(row_hz / 30))
  with video.writer(args.out, 3 * TILE, TILE, row_hz / stride * args.speed) as write:
    for row in range(0, len(rows.timestamp), stride):
      data.qpos[:] = rows.qpos[row]
      mujoco.mj_kinematics(env.model, data)
      overview.update_scene(data, camera, scene_option=shown)
      images, _ = env.trace(rows.qpos[row][None])
      write(np.concatenate([overview.render(), *images[0]], axis=1))
  print(f"{args.out}: {rows.timestamp[-1]:.1f} s of {recorded.meta['task']}")
