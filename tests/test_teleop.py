"""Teleoperation: the headset's frames, the controller's binding, the page's
socket, and a recorded episode driven by a controller."""

import json
import socket
import time
import urllib.request
from collections.abc import Callable
from typing import Any

import numpy as np
from websockets.sync.client import connect

from specialist_sim import episode, rotation
from specialist_sim.rig import Clock
from specialist_sim.sort_lego import Cfg, SortLegoEnv, scene
from specialist_sim.sort_lego.initial import MODES
from specialist_sim.teleop import binding, server, xr

FOCUS = np.array([*scene.TRAY_CENTER, scene.DESK_TOP])
FLOOR = scene.DESK_TOP - 0.75


def _pose(rng: np.random.Generator) -> np.ndarray:
  quaternion = rng.normal(size=4)
  return np.concatenate(
    [rng.uniform(-1, 1, 3), quaternion / np.linalg.norm(quaternion)]
  )


def test_an_anchor_takes_a_headset_pose_back_to_the_room() -> None:
  rng = np.random.default_rng(0)
  head = np.concatenate([[0.3, 1.7, -0.2], rotation.exp(np.array([0.1, 0.8, -0.05]))])
  anchor = xr.facing(head, FOCUS, FLOOR)
  room = _pose(rng)
  matrix = anchor.matrix()
  headset = np.concatenate(
    [
      matrix[:3, :3] @ room[:3] + matrix[:3, 3],
      rotation.multiply(rotation.from_matrix(matrix[:3, :3]), room[3:]),
    ]
  )
  back = anchor.to_room(headset)
  assert np.allclose(back[:3], room[:3], atol=1e-9)
  assert abs(abs(np.dot(back[3:], room[3:])) - 1.0) < 1e-9


def test_facing_stands_the_focus_ahead_and_the_floor_on_the_real_one() -> None:
  head = np.concatenate([[0.4, 1.65, 0.1], rotation.exp(np.array([0.0, 0.7, 0.0]))])
  anchor = xr.facing(head, FOCUS, FLOOR)
  focus = anchor.rotation @ FOCUS + anchor.offset
  ahead = rotation.rotate(head[3:], np.array([0.0, 0.0, -1.0]))
  assert np.allclose(focus[[0, 2]], head[[0, 2]] + xr.REACH * ahead[[0, 2]], atol=1e-9)
  assert abs(focus[1] - (FOCUS[2] - FLOOR)) < 1e-9
  # The operator faces the bins: the room's -x is the way the head faces.
  assert np.allclose(anchor.rotation @ np.array([-1.0, 0.0, 0.0]), ahead, atol=1e-9)


def test_a_squeeze_binds_without_a_jump_and_the_next_releases() -> None:
  rng = np.random.default_rng(1)
  grab = binding.Grab()
  controller, target = _pose(rng), np.concatenate([_pose(rng), [0.2]])
  assert np.array_equal(grab.update(controller, 0.0, 0.0, target), target)
  bound = grab.update(controller, 1.0, 0.0, target)
  assert np.allclose(bound[:7], target[:7], atol=1e-9)
  assert bound[7] == binding.OPEN and grab.hold is not None
  grab.update(controller, 0.5, 0.0, bound)
  assert grab.hold is not None, "a half-held grip does not release"
  turn = rotation.exp(np.array([0.0, 0.0, 0.5]))
  moved = np.concatenate(
    [controller[:3] + [0.1, 0.0, 0.0], rotation.multiply(turn, controller[3:])]
  )
  carried = grab.update(moved, 0.0, 1.0, bound)
  lever = target[:3] - controller[:3]
  assert np.allclose(carried[:3], moved[:3] + rotation.rotate(turn, lever), atol=1e-9)
  assert carried[7] == binding.SHUT
  assert np.array_equal(grab.update(None, 0.0, 1.0, carried), carried)
  assert grab.hold is not None, "an untracked controller keeps its binding"
  released = grab.update(moved, 1.0, 0.0, carried)
  assert np.array_equal(released, carried) and grab.hold is None


def test_the_sticks_move_the_scene_the_way_the_operator_faces() -> None:
  head = np.concatenate(
    [[0.0, 1.6, 0.0], rotation.exp(np.array([0.0, np.pi / 2, 0.0]))]
  )
  anchor = xr.facing(head, FOCUS, FLOOR)

  def focus(a: xr.Anchor) -> np.ndarray:
    out: np.ndarray = a.rotation @ FOCUS + a.offset
    return out

  pushed = xr.steer(anchor, head, FOCUS, (0.0, -1.0), (0.0, 0.0), dt=1.0)
  assert np.allclose(focus(pushed) - focus(anchor), [-xr.SLIDE, 0.0, 0.0], atol=1e-9)
  raised = xr.steer(anchor, head, FOCUS, (0.0, 0.0), (0.0, -1.0), dt=1.0)
  assert np.allclose(focus(raised) - focus(anchor), [0.0, xr.RAISE, 0.0], atol=1e-9)
  still = xr.steer(anchor, head, FOCUS, (0.05, 0.05), (0.05, 0.05), dt=1.0)
  assert np.array_equal(still.matrix(), anchor.matrix())
  turned = xr.moved(anchor, np.zeros(3), np.radians(20.0), FOCUS)
  assert np.allclose(focus(turned), focus(anchor), atol=1e-12), (
    "it turns about the focus"
  )
  stepped = xr.nudged(
    xr.facing(np.array([0, 1.6, 0, 1, 0, 0, 0.0]), FOCUS, FLOOR),
    None,
    FOCUS,
    0.01,
    0.02,
    0.03,
    0.0,
  )
  start = xr.facing(np.array([0, 1.6, 0, 1, 0, 0, 0.0]), FOCUS, FLOOR)
  assert np.allclose(focus(stepped) - focus(start), [0.02, 0.03, -0.01], atol=1e-12)


def _free_port() -> int:
  with socket.socket() as probe:
    probe.bind(("localhost", 0))
    port: int = probe.getsockname()[1]
    return port


def _gather(
  drain: Callable[[], list[Any]], count: int, seconds: float = 2.0
) -> list[Any]:
  out: list[Any] = []
  deadline = time.perf_counter() + seconds
  while len(out) < count and time.perf_counter() < deadline:
    out += drain()
    time.sleep(0.01)
  return out


def test_the_console_and_the_headset_round_trip() -> None:
  # The clear bins wear the scene's only textures, their cards.
  env = SortLegoEnv(Cfg(render=False, bins="clear"))
  env.reset([0])
  port = _free_port()
  console = server.Page(env.rig.model, port)
  console.stage(
    env.rig.datas[0],
    -1.14,
    FOCUS,
    FOCUS + 0.4,
    MODES,
    {"mode": "single", "bricks": None},
  )
  html = urllib.request.urlopen(f"http://localhost:{port}/").read().decode()
  assert "renderer.xr" in html and 'data-command="record"' in html
  with connect(f"ws://localhost:{port}/", max_size=None) as browser:
    staged = json.loads(browser.recv())
    assert staged["type"] == "scene" and len(staged["geoms"]) == len(console.drawn) > 0
    assert staged["modes"] == list(MODES) and staged["floor"] == -1.14
    assert sum(geom.get("texture") is not None for geom in staged["geoms"]) == 2
    assert {geom["hand"] for geom in staged["geoms"]} == {-1, 0, 1}
    for buttons in ([False] * 6, [False, True, False, False, True, False]):
      hand = {"pose": [0.1, 1.0, -0.3, 1, 0, 0, 0], "trigger": 0.25, "squeeze": 1.0}
      hand |= {"buttons": buttons, "axes": [0.0, 0.0, 0.5, -0.75]}
      browser.send(
        json.dumps(
          {"type": "input", "head": [0, 1.6, 0, 1, 0, 0, 0], "hands": {"right": hand}}
        )
      )
    browser.send(json.dumps({"type": "command", "name": "drop"}))
    browser.send(json.dumps({"type": "layout", "mode": "pile", "bricks": 4}))
    assert _gather(console.commands, 2) == ["record", "drop"]
    assert _gather(console.layouts, 1) == [{"mode": "pile", "bricks": 4}]
    latest = console.latest()
    assert latest is not None and latest.hands["right"].stick == (0.5, -0.75)
    browser.send(json.dumps({"type": "nudge", "forward": 0.01, "turn": 0.02}))
    assert _gather(console.nudges, 1) == [server.Nudge(forward=0.01, turn=0.02)]
    console.publish(
      env.rig.datas[0], np.eye(4), {"recording": False}, {"head": np.arange(7.0)}
    )
    frame = json.loads(browser.recv())
    assert frame["type"] == "frame" and frame["head"] == list(range(7))
    console.show({"right": np.zeros((8, 8, 3), np.uint8)})
    picture = browser.recv()
    while isinstance(picture, str):
      picture = browser.recv()
    assert (
      picture[1 : 1 + picture[0]] == b"right"
      and picture[1 + picture[0] :][:2] == b"\xff\xd8"
    )


def test_a_bound_controller_carries_a_hand_and_every_row_is_recorded() -> None:
  clock = Clock(physics_dt=1.0 / 1500.0, per_row=10, per_action=1, per_step=1)
  env = SortLegoEnv(Cfg(clock=clock, render=False, bricks=1))
  observation, _ = env.reset([3])
  target = observation.state[0, -1].copy()
  grab = binding.Grab()
  controller = np.array([-0.2, 0.2, -0.1, 1.0, 0.0, 0.0, 0.0])
  recorder = episode.Recorder()
  recorder.start("quest", {"source": "quest", **env.describe(0)})
  recorder.extend(*(rows[-1:] for rows in env.rig.rows(0)))
  for tick in range(150):
    controller[0] -= 0.0003
    target[1] = grab.update(controller, 1.0 if tick == 0 else 0.0, 0.0, target[1])
    observation, _ = env.step(target[None, None])
    recorder.extend(*env.rig.rows(0))
  recorded = recorder.finish()
  assert len(recorded.timestamp) == 151
  assert np.allclose(np.diff(recorded.timestamp), clock.row_dt)
  moved = recorded.state[-1, 1, 0] - recorded.state[0, 1, 0]
  assert abs(moved + 150 * 0.0003) < 0.01, "the hand followed the controller"
  assert np.allclose(recorded.commanded[-1, 1, :3], target[1, :3])
