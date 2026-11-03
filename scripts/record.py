"""Record LEGO sorting episodes handheld: a Quest carries the grippers,
a browser tab is the console. This is the first-choice way to collect
demonstrations in simulation.

  uv run --extra handheld python scripts/record.py --out ~/recordings/NAME

The console is http://localhost:8012 and the headset opens the same page,
then Enter VR or Enter AR. With the headset on USB, the run keeps the adb
reverse in place. Over Wi-Fi, serve with ``--cert`` and ``--key``. Each gripper
rides its controller, held by its handle.

    trigger           close the jaws
    A  (console R)    paused: record. Recording: keep the episode, lay a new tray, record it
    B  (console D)    drop the episode, lay a new tray, and pause
    X  (console C)    stand the scene in front of you again
    left stick        slide the scene (console: arrow keys)
    right stick       raise the scene (up, down), turn it (left, right) (console: PgUp, PgDn, Q, E)

A new tray keeps the grippers where they stand, and a
layout that would put a brick in a gripper is passed over. Every row of an
episode is recorded.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from specialist_sim import episode
from specialist_sim.arrays import Float64Array
from specialist_sim.env import Cfg, Env
from specialist_sim.handheld import mount, server
from specialist_sim.handheld.server import HANDS, Inputs, Page
from specialist_sim.handheld.xr import ahead, facing, nudged, steer
from specialist_sim.rig import Clock
from specialist_sim.sort_lego import SortLego, scene
from specialist_sim.sort_lego.initial import MODES

CLOCK = Clock(per_action=1, per_step=1)
"""One row a step, the controllers' pose its one setpoint."""
STANDING = np.array([0.0, 1.6, 0.0, 1.0, 0.0, 0.0, 0.0])
"""A head the scene is stood before until the headset reports its own."""
PIVOT = np.array([*scene.TRAY_CENTER, scene.DESK_TOP])
"""What the scene is stood before the demonstrator by and turned about: the
tray's middle, so the tray stays in front."""
FOCUS = np.array([scene.TRAY_CENTER[0] - 0.08, scene.TRAY_CENTER[1], scene.DESK_TOP])
"""What the console's desktop view looks at."""
PANEL = np.array(
  [
    scene.DESK_EDGE - 0.05,
    scene.DESK_CENTER[1],
    scene.DESK_TOP + 0.40,
  ]
)
"""Where the status panel stands: behind the bins, above the demonstrator's view
of them."""
FRESH = 0.05
"""The most two headset frames may stand apart, seconds, for the newer to be
carried on along their motion. The headset draws at 72 Hz or more and the loop
runs at the row rate: a pose held until the next frame would step the setpoint,
its speed twice the hand's and then none."""


def carried(
  newest: Inputs, older: Inputs | None, hand: str, now: float
) -> Float64Array:
  """A controller's pose in headset space at ``now``, ``time.perf_counter()``:
  the newest frame's, carried on along its motion since the frame before, by the
  time since it arrived (at most a frame's)."""
  pose = newest.hands[hand].pose
  if (
    older is None
    or hand not in older.hands
    or newest.drawn is None
    or older.drawn is None
  ):
    return pose
  apart = newest.drawn - older.drawn
  if not 0.0 < apart <= FRESH:
    return pose
  return ahead(pose, older.hands[hand].pose, apart, min(now - newest.at, apart))


def main() -> None:
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
    "--out", type=Path, required=True, help="where the episodes are written"
  )
  parser.add_argument(
    "--mode", choices=MODES, default=SortLego.mode, help="how a tray is laid out"
  )
  parser.add_argument(
    "--bricks",
    type=int,
    default=SortLego.bricks,
    help="bricks on a tray (default: the mode's count)",
  )
  parser.add_argument(
    "--seed", type=int, default=None, help="the first tray's seed (default: the clock)"
  )
  parser.add_argument("--port", type=int, default=server.PORT)
  parser.add_argument(
    "--cert", type=Path, default=None, help="a TLS certificate, to serve over Wi-Fi"
  )
  parser.add_argument("--key", type=Path, default=None, help="its private key")
  args = parser.parse_args()

  task = SortLego(mode=args.mode, bricks=args.bricks)
  env = Env(task, Cfg(clock=CLOCK, render=False))
  model, data = env.rig.model, env.rig.datas[0]
  seed = int(time.time()) if args.seed is None else args.seed
  observation, _ = env.reset([seed])
  console = Page(model, args.port, args.cert, args.key)
  console.stage(data, FOCUS, PANEL, MODES, args.mode)
  print(f"the console is http://localhost:{args.port}; the headset opens the same page")
  server.keep_reversed(args.port)

  anchor, centered = facing(STANDING, PIVOT), False
  newest: Inputs | None = None
  older: Inputs | None = None
  target = observation.state[0, -1].copy()
  recorder = episode.Recorder()
  kept = len(episode.listing(args.out))
  note = "A to record"

  def new_tray() -> None:
    """New bricks under the hands as they stand: their poses and what they were
    sent carry over."""
    nonlocal seed
    env.reset([seed + 1], start=env.rig.state_rows[0, -1])
    layout = task.layouts[0]
    assert layout is not None
    seed = layout.seed

  def start() -> None:
    recorder.start(
      time.strftime("%Y%m%d_%H%M%S"), {"source": "quest", **env.describe()}
    )
    recorder.extend(*(rows[-1:] for rows in env.rig.rows(0)))
    console.pulse("right")

  period, shown = CLOCK.row_dt, 0.0
  due = time.perf_counter()
  while True:
    inputs = console.latest()
    if inputs is not None and (newest is None or inputs.at != newest.at):
      older, newest = newest, inputs
    if inputs is not None and not centered:
      anchor, centered = facing(inputs.head, PIVOT), True
    for asked in console.layouts():
      if asked in MODES:
        # A tray's mode is read only when it is laid.
        task.mode = asked
        note = f"next trays: {asked}"
    for command in console.commands():
      if command == "record" and recorder.name is None:
        note = "recording"
        start()
      elif command == "record":
        score = task.score(0)
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
        anchor = facing(inputs.head, PIVOT)

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
    now = time.perf_counter()
    for index, hand in enumerate(HANDS):
      if inputs is None or hand not in inputs.hands:
        continue  # an untracked controller's gripper stands where it was sent
      room[hand] = anchor.to_room(carried(inputs, older, hand, now))
      target[index] = mount.target(room[hand], inputs.hands[hand].trigger)
    observation, _ = env.step(target[None, None])
    recorder.extend(*env.rig.rows(0))
    standing = observation.state[0, -1]

    now = time.perf_counter()
    if now - shown >= 1.0 / server.RATE:
      shown = now
      recording = recorder.name is not None
      tracked = [] if inputs is None else sorted(inputs.hands)
      status = {
        "recording": recording,
        "headline": f"REC {len(recorder) * period:5.1f} s" if recording else "Paused",
        "detail": f"{task.mode} tray {seed} · {kept} kept · {note}",
        "headset": "not connected"
        if inputs is None
        else f"tracking {', '.join(tracked) or 'no controller'}",
        "hands": {
          hand: {"opening": float(standing[index, 7])}
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
