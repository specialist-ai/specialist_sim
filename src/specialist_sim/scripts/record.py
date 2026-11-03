"""Record episodes of a task handheld: a Quest carries the grippers, the
headset's browser shows the scene.

  uv run --extra handheld sim-record sort-lego --out ~/recordings/NAME
  uv run --extra handheld sim-record pick-place --out ~/recordings/NAME

Each task's fields are its flags, ``sim-record sort-lego --mode pile``.

The headset opens http://localhost:8012, then Enter VR or Enter AR. With the
headset on USB, the run keeps the adb reverse in place. Over Wi-Fi, serve with
``--cert`` and ``--key``. Each gripper rides its controller, held by its handle.
The left controller starts and the right one ends:

    trigger    close the jaws
    X          paused: record
    Y          stand the scene in front of you again
    A          recording: keep the episode, lay a new layout, and pause
    B          recording: drop the episode and record a new layout

A new layout keeps the grippers where they stand, and one that would put an
object in a gripper is passed over. Every row of an episode is recorded.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro

from specialist_sim import episode
from specialist_sim.env import ROWWISE, Cfg, Env, Start
from specialist_sim.episode import Episode, Rows
from specialist_sim.gripper import HANDS
from specialist_sim.handheld import server, xr
from specialist_sim.handheld.server import Page
from specialist_sim.handheld.xr import Inputs
from specialist_sim.tasks.pick_place import PickPlace
from specialist_sim.tasks.sort_lego import SortLego

Recorded = (
  Annotated[SortLego, tyro.conf.subcommand("sort-lego")]
  | Annotated[PickPlace, tyro.conf.subcommand("pick-place")]
)

NOTICE = 2.0
RED, GREY = "rgba(190, 25, 20, 0.88)", "rgba(30, 30, 30, 0.8)"
GREEN = "rgba(25, 135, 60, 0.9)"


def main() -> None:
  tyro.cli(
    record,
    description=__doc__,
    config=(tyro.conf.CascadeSubcommandArgs, tyro.conf.OmitSubcommandPrefixes),
  )


def record(
  task: Recorded,
  out: Path,
  port: int = 8012,
  cert: Path | None = None,
  key: Path | None = None,
) -> None:
  """Record episodes of ``task`` into ``out``.

  Args:
    task: The task.
    out: Where the episodes are written.
    port: The port the headset opens.
    cert: A TLS certificate, to serve over Wi-Fi.
    key: Its private key.
  """
  env = Env(task, Cfg(clock=ROWWISE, finger=xr.FINGER, image_hands=()))
  focus = env.task.focus
  seed = int(time.time())
  observation, info = env.reset([seed])
  page = Page(
    env.model, env.sim.geom_xpos[0], env.sim.geom_xmat[0], focus, port, cert, key
  )
  print(f"the headset opens http://localhost:{port}")
  server.keep_reversed(port)

  # The first headset frame centers the scene, then each recenter does.
  anchor = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
  centered = False
  newest: Inputs | None = None
  older: Inputs | None = None
  sent = observation.state[0, -1].copy()
  taken: list[Rows] | None = None
  name = ""
  meta = {}
  kept, total = 0, 0.0
  period = ROWWISE.row_dt
  notice = ("", GREY, 0.0)

  def notify(headline: str, color: str) -> None:
    nonlocal notice
    notice = (headline, color, time.perf_counter() + NOTICE)

  def new_layout() -> None:
    """A new layout under the hands as they stand."""
    nonlocal seed, info
    _, info = env.reset([seed + 1], start=Start(env.rows(0).state[-1]))
    seed = int(info["seed"][0])

  def start() -> None:
    nonlocal taken, name, meta
    taken = [env.rows(0)]
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    meta = env.describe()

  due = time.perf_counter()
  while True:
    poll = page.poll()
    inputs = poll.inputs
    if inputs is not None and (newest is None or inputs.at != newest.at):
      older, newest = newest, inputs
    if inputs is not None and not centered:
      anchor, centered = xr.facing(inputs.head, focus), True
    for command in poll.commands:
      if command == "record" and taken is None:
        start()
      elif command == "keep" and taken is not None:
        counts = {key: int(value[0]) for key, value in info["counts"].items()}
        rows = episode.joined(taken)
        path = episode.save(out, Episode(name, rows, {**meta, "score": counts}))
        kept += 1
        total += len(taken) * period
        taken = None
        notify(f"Kept #{kept}", GREEN)
        print(f"kept {path.stem}, progress {info['progress'][0]:.2f}")
        new_layout()
      elif command == "drop" and taken is not None:
        notify("Dropped", GREY)
        new_layout()
        start()
      elif command == "recenter" and inputs is not None:
        anchor = xr.facing(inputs.head, focus)

    if inputs is not None:
      now = time.perf_counter()
      # An untracked controller's gripper stands where it was last sent.
      for index, hand in enumerate(HANDS):
        if hand in inputs.hands:
          placed = xr.to_room(anchor, xr.carried(inputs, older, hand, now))
          sent[index] = xr.target(placed, inputs.hands[hand].trigger)
    _, info = env.step(sent[None, None])
    if taken is not None:
      taken.append(env.rows(0))

    headline, color, until = notice
    if time.perf_counter() >= until:
      headline, color = "Paused", GREY
      if taken is not None:
        headline, color = f"REC {len(taken) * period:5.1f} s", RED
    status = {
      "headline": headline,
      "color": color,
      "detail": f"{kept} kept · {total / 60:.1f} min",
      "progress": float(info["progress"][0]),
      "buttons": "X record · Y recenter" if taken is None else "A keep · B drop",
      "ended": until,
    }
    page.show(env.sim.geom_xpos[0], env.sim.geom_xmat[0], anchor, status)
    due += period
    wait = due - time.perf_counter()
    if wait > 0:
      time.sleep(wait)
    elif wait < -0.1:  # fell behind: the clock restarts from now
      due = time.perf_counter()
