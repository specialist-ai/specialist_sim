"""Film a recorded LEGO sorting episode, teleoperated, scripted or evaluated:
an overview of the desk beside each hand's wrist lens, traced from the
recorded states at 30 Hz.

  MUJOCO_GL=egl uv run python scripts/render_video.py runs/NAME/drop_200003.npz \\
    --out videos/drop_200003.mp4

The lenses are the whole 720 px frame halved, not the views a policy crops
from it. The overview is MuJoCo's own renderer looking at the tray from the
operator's side, drawing the geom groups the lenses draw, so no collider shows.
It needs ``ffmpeg`` and a GPU.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from specialist_sim import episode
from specialist_sim.gripper import SLOTS
from specialist_sim.render import DRAWN_GROUPS, Renderer
from specialist_sim.sort_lego import Cfg, SortLegoEnv, scene
from specialist_sim.sort_lego.env import LENS

FPS = 30
SIDE = 360
"""Pixels across each panel."""


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
  meta = recorded.meta
  cfg = Cfg(
    render=False,
    colors=tuple(meta["colors"]),
    swapped=meta.get("swapped", False),
    bins=meta.get("bins", "clear"),
  )
  rig = SortLegoEnv(cfg).rig
  model, data = rig.model, mujoco.MjData(rig.model)
  lenses = Renderer(
    model, {hand.slot: (hand.base, LENS) for hand in rig.hands}, 2 * SIDE
  )
  overview = mujoco.Renderer(model, SIDE, SIDE)
  shown = mujoco.MjvOption()
  shown.geomgroup[:] = [group in DRAWN_GROUPS for group in range(len(shown.geomgroup))]
  camera = mujoco.MjvCamera()
  camera.lookat[:] = [scene.TRAY_CENTER[0] - 0.08, scene.TRAY_CENTER[1], scene.DESK_TOP]
  camera.distance, camera.azimuth, camera.elevation = 0.9, 180.0, -50.0
  args.out.parent.mkdir(parents=True, exist_ok=True)
  encoder = subprocess.Popen(
    ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3 * SIDE}x{SIDE}",
     "-r", str(FPS * args.speed), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
     str(args.out)],
    stdin=subprocess.PIPE,
  )  # fmt: skip
  assert encoder.stdin is not None
  stride = round(meta["row_hz"] / FPS)
  for row in range(0, len(recorded.timestamp), stride):
    data.qpos[:] = recorded.qpos[row]
    mujoco.mj_kinematics(model, data)
    overview.update_scene(data, camera, scene_option=shown)
    images, _ = lenses.render_many([data], list(SLOTS))
    frame = np.concatenate([overview.render(), *images[0, :, ::2, ::2]], axis=1)
    encoder.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
  encoder.stdin.close()
  encoder.wait()
  score = meta.get("score", {})
  print(
    f"{args.out}: {recorded.timestamp[-1]:.1f} s, "
    f"{score.get('sorted', '?')}/{score.get('started', '?')} sorted"
  )


if __name__ == "__main__":
  main()
