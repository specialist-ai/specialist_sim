"""Record LEGO sorting episodes by teleoperation: a Quest carries the grippers,
a browser tab is the console. This is the first-choice way to collect
demonstrations in simulation, and ``scripts/record_scripted.py`` is the fallback.

  uv run --extra teleop python scripts/record.py --out data/episodes/NAME

Open http://localhost:8012 on this machine for the console. Open the same
address in the headset's browser and press Enter VR or Enter AR. With the
headset on USB, the run keeps the adb reverse in place. Over Wi-Fi, serve with
``--cert`` and ``--key``.

    grip, squeezed    bind the gripper on that side, squeeze again to release it
    trigger           close the jaws of a bound gripper
    A  (console R)    paused: record. Recording: keep the episode, lay a new tray, record it
    B  (console D)    drop the episode, lay a new tray, and pause
    X  (console C)    stand the scene in front of you again
    left stick        slide the scene (console: arrow keys)
    right stick       raise the scene (up, down), turn it (left, right) (console: PgUp, PgDn, Q, E)

A new tray keeps the grippers where they stand and keeps their bindings, and a
layout that would put a brick in a gripper is passed over. ``--lenses`` shows
both wrist lenses live (GPU). Every 150 Hz row of an episode is recorded.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from specialist_sim import episode
from specialist_sim.gripper import SLOTS
from specialist_sim.render import Renderer
from specialist_sim.rig import Clock
from specialist_sim.sort_lego import Cfg, SortLegoEnv, scene
from specialist_sim.sort_lego.env import LENS
from specialist_sim.sort_lego.initial import MODES
from specialist_sim.teleop import server
from specialist_sim.teleop.binding import Grab
from specialist_sim.teleop.server import HANDS, Page
from specialist_sim.teleop.xr import facing, nudged, steer

CLOCK = Clock(physics_dt=1.0 / 1500.0, per_row=10, per_action=1, per_step=1)
"""One 150 Hz row a step, the controllers' pose its one setpoint."""
STANDING = np.array([0.0, 1.6, 0.0, 1.0, 0.0, 0.0, 0.0])
"""A head the scene is stood before until the headset reports its own."""
PIVOT = np.array([*scene.TRAY_CENTER, scene.DESK_TOP])
"""What the scene is stood before the operator by and turned about: the
tray's middle, so the tray stays in front."""
FOCUS = np.array([scene.TRAY_CENTER[0] - 0.08, scene.TRAY_CENTER[1], scene.DESK_TOP])
"""What the console's desktop view looks at."""
PANEL = np.array(
  [
    scene.BIN_NEAR_FACE - scene.BIN_SIZE[0] - 0.05,
    scene.DESK_CENTER[1],
    scene.DESK_TOP + 0.40,
  ]
)
"""Where the status panel stands: behind the bins, above the operator's view
of them."""
LENS_RATE = 10.0
LENS_SIZE = 240


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--out", type=Path, required=True, help="where the episodes are written"
  )
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
    "--seed", type=int, default=None, help="the first tray's seed (default: the clock)"
  )
  parser.add_argument(
    "--desk-height",
    type=float,
    default=scene.DESK_HEIGHT,
    help="the real desk's top over the floor, m",
  )
  parser.add_argument("--port", type=int, default=server.PORT)
  parser.add_argument(
    "--cert", type=Path, default=None, help="a TLS certificate, to serve over Wi-Fi"
  )
  parser.add_argument("--key", type=Path, default=None, help="its private key")
  parser.add_argument(
    "--lenses", action="store_true", help="show each wrist lens live (GPU)"
  )
  args = parser.parse_args()

  env = SortLegoEnv(Cfg(clock=CLOCK, render=False, mode=args.mode, bricks=args.bricks))
  model, data = env.rig.model, env.rig.datas[0]
  seed = int(time.time()) if args.seed is None else args.seed
  observation, _ = env.reset([seed])
  # The real floor in the room frame: the desk top stands desk_height over it.
  floor = scene.DESK_TOP - args.desk_height
  console = Page(model, args.port, args.cert, args.key)
  console.stage(
    data, floor, FOCUS, PANEL, MODES, {"mode": args.mode, "bricks": args.bricks}
  )
  previews = None
  if args.lenses:
    cameras = {hand.slot: (hand.base, LENS) for hand in env.rig.hands}
    previews = Renderer(model, cameras, LENS_SIZE)
  print(f"the console is http://localhost:{args.port}; the headset opens the same page")
  server.keep_reversed(args.port)

  anchor, centered = facing(STANDING, PIVOT, floor), False
  grabs = {hand: Grab() for hand in HANDS}
  target = observation.state[0, -1].copy()
  recorder = episode.Recorder()
  kept = len(episode.listing(args.out))
  note = "A to record"

  def new_tray() -> None:
    """New bricks under the hands as they stand: their poses, their bindings and
    what they were sent all carry over."""
    nonlocal seed
    env.reset([seed + 1], hands=env.rig.state_rows[0, -1])
    layout = env.layouts[0]
    assert layout is not None
    seed = layout.seed

  def start() -> None:
    recorder.start(
      time.strftime("%Y%m%d_%H%M%S"), {"source": "quest", **env.describe(0)}
    )
    recorder.extend(*(rows[-1:] for rows in env.rig.rows(0)))
    console.pulse("right")

  period, shown, imaged = CLOCK.row_dt, 0.0, 0.0
  due = time.perf_counter()
  while True:
    inputs = console.latest()
    if inputs is not None and not centered:
      anchor, centered = facing(inputs.head, PIVOT, floor), True
    for asked in console.layouts():
      if asked["mode"] in MODES:
        # A tray's mode and count are read only when it is laid.
        env.cfg = replace(env.cfg, mode=asked["mode"], bricks=asked["bricks"])
        note = (
          f"next trays: {asked['mode']}, {asked['bricks'] or 'its own count of'} bricks"
        )
    for command in console.commands():
      if command == "record" and recorder.name is None:
        note = "recording"
        start()
      elif command == "record":
        score = env.score(0)
        path = episode.save(args.out, recorder.finish({"score": score.as_dict()}))
        kept += 1
        note = (
          f"kept {path.stem}: {score.sorted}/{score.started} sorted. Recording the next"
        )
        print(note)
        new_tray()
        start()
      elif command == "drop":
        note = "a new tray. A to record it"
        if recorder.name is not None:
          recorder.discard()
          note = "dropped the episode. A to record this tray"
          print(note)
        new_tray()
      elif command == "recenter" and inputs is not None:
        anchor = facing(inputs.head, PIVOT, floor)

    for step in console.nudges():
      head = None if inputs is None else inputs.head
      anchor = nudged(anchor, head, PIVOT, step.forward, step.right, step.up, step.turn)
    if inputs is not None:
      sticks = {hand: held.stick for hand, held in inputs.hands.items()}
      still = (0.0, 0.0)
      anchor = steer(
        anchor,
        inputs.head,
        PIVOT,
        sticks.get("left", still),
        sticks.get("right", still),
        period,
      )

    room = {} if inputs is None else {"head": anchor.to_room(inputs.head)}
    for index, hand in enumerate(HANDS):
      held = None if inputs is None else inputs.hands.get(hand)
      pose = None if held is None else anchor.to_room(held.pose)
      if pose is not None:
        room[hand] = pose
      squeeze, trigger = (0.0, 0.0) if held is None else (held.squeeze, held.trigger)
      target[index] = grabs[hand].update(pose, squeeze, trigger, target[index])
    observation, _ = env.step(target[None, None])
    recorder.extend(*env.rig.rows(0))
    standing = observation.state[0, -1]

    now = time.perf_counter()
    if previews is not None and now - imaged >= 1.0 / LENS_RATE:
      imaged = now
      images, _ = previews.render_many([data], list(SLOTS))
      console.show(dict(zip(HANDS, images[0], strict=True)))
    if now - shown >= 1.0 / server.RATE:
      shown = now
      recording = recorder.name is not None
      tracked = [] if inputs is None else sorted(inputs.hands)
      status = {
        "recording": recording,
        "headline": f"REC {len(recorder) * period:5.1f} s" if recording else "Paused",
        "detail": f"{env.cfg.mode} tray {seed} · {kept} kept · {note}",
        "headset": "not connected"
        if inputs is None
        else f"tracking_error {', '.join(tracked) or 'no controller'}",
        "hands": {
          hand: {
            "held": grabs[hand].hold is not None,
            "opening": float(standing[index, 7]),
          }
          for index, hand in enumerate(HANDS)
        },
      }
      console.publish(data, anchor.matrix(), status, room)
    due += period
    wait = due - time.perf_counter()
    if wait > 0:
      time.sleep(wait)
    elif wait < -0.1:  # fell behind: carry on from now rather than race to catch up
      due = time.perf_counter()


if __name__ == "__main__":
  main()
