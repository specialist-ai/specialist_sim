"""The headset's frames, the gripper's mount, and the page's socket."""

import json
import socket
import time
import urllib.request
from collections.abc import Callable

import numpy as np
from websockets.sync.client import connect

from fixtures.scene import BLIND
from specialist_sim import gripper, rotation
from specialist_sim.arrays import Float64Array
from specialist_sim.camera.render import parts
from specialist_sim.env import Env
from specialist_sim.handheld import server, xr
from specialist_sim.tasks.sort_lego import SortLego

FOCUS = SortLego.focus


def _placed(anchor: Float64Array, point: Float64Array) -> Float64Array:
  """A room point in headset space."""
  return anchor[:3] + rotation.rotate(anchor[3:], point)


def test_facing_stands_the_focus_ahead_and_the_room_on_the_real_floor() -> None:
  head = np.concatenate([[0.4, 1.65, 0.1], rotation.exp(np.array([0.0, 0.7, 0.0]))])
  anchor = xr.facing(head, FOCUS)
  focus = _placed(anchor, FOCUS)
  ahead = rotation.rotate(head[3:], np.array([0.0, 0.0, -1.0]))
  assert np.allclose(focus[[0, 2]], head[[0, 2]] + xr.REACH * ahead[[0, 2]], atol=1e-9)
  assert abs(focus[1] - FOCUS[2]) < 1e-9
  # The demonstrator faces the bins: the room's +x is the way the head faces.
  assert np.allclose(
    rotation.rotate(anchor[3:], np.array([1.0, 0.0, 0.0])), ahead, atol=1e-9
  )
  back = xr.to_room(anchor, np.concatenate([focus, head[3:]]))
  assert np.allclose(back[:3], FOCUS, atol=1e-9)


def test_the_left_controller_lands_on_the_left_hands_side_and_ahead_is_plus_x() -> None:
  head = np.concatenate([[0.4, 1.65, 0.1], rotation.exp(np.array([0.0, -1.2, 0.0]))])
  anchor = xr.facing(head, FOCUS)

  def room(offset: Float64Array) -> Float64Array:
    """A point ``offset`` from the head in its own axes, in the room."""
    at = head[:3] + rotation.rotate(head[3:], offset)
    return xr.to_room(anchor, np.concatenate([at, head[3:]]))[:3]

  left = room(np.array([-0.25, -0.4, -0.3]))  # the left controller, low and ahead
  assert np.sign(left[1]) == gripper.ACROSS["left"] and left[1] > 0.0
  assert room(np.array([0.0, 0.0, -1.0]))[0] > room(np.zeros(3))[0] + 0.99


def test_a_pose_carried_ahead_keeps_its_speed_and_turn() -> None:
  turn = rotation.exp(np.array([0.0, 0.2, 0.0]))
  older = np.array([0.1, 1.0, -0.3, 1.0, 0.0, 0.0, 0.0])
  newer = np.concatenate([older[:3] + [0.02, 0.0, 0.0], turn])

  def frame(pose: Float64Array, drawn: float) -> xr.Inputs:
    hands = {"right": xr.Controller(pose, 0.0)}
    return xr.Inputs(older, hands, at=drawn, drawn=drawn)

  before, after = frame(older, 1.0), frame(newer, 1.014)
  carried = xr.carried(after, before, "right", 1.021)
  assert np.allclose(carried[:3], older[:3] + [0.03, 0.0, 0.0])
  assert np.allclose(carried[3:], rotation.exp(np.array([0.0, 0.3, 0.0])))
  assert np.allclose(xr.carried(after, before, "right", after.at), newer)


def test_the_mount_holds_the_gripper_by_its_handle() -> None:
  """A controller held upright in a right hand, pointing ahead along -z, in
  WebXR's own axes: the back of the hand toward +x, the arm toward +z."""
  # The rod's -z turned to +y, up.
  upright = rotation.exp(np.array([np.pi / 2, 0.0, 0.0]))
  controller = np.array([0.3, 1.0, -0.4, *upright])
  tag = xr.target(controller, 0.0)
  assert np.allclose(tag[:3], [0.3, 1.0 + xr.HANDLE, -0.4]), "above the fist"
  assert tag[7] == gripper.OPEN and xr.target(controller, 1.0)[7] == gripper.SHUT
  axes = [rotation.rotate(tag[3:7], axis) for axis in np.eye(3)]
  assert np.allclose(axes, [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], atol=1e-12), (
    "forward ahead, left to the left, up the handle"
  )


def _free_port() -> int:
  with socket.socket() as probe:
    probe.bind(("localhost", 0))
    port: int = probe.getsockname()[1]
    return port


def _polled(ready: Callable[[server.Poll], bool], page: server.Page) -> server.Poll:
  """The polls of two seconds merged, or until ``ready``."""
  deadline = time.perf_counter() + 2.0
  merged = server.Poll(None, [])
  while not ready(merged) and time.perf_counter() < deadline:
    poll = page.poll()
    merged = server.Poll(poll.inputs or merged.inputs, merged.commands + poll.commands)
    time.sleep(0.01)
  return merged


def test_the_page_and_the_headset_round_trip() -> None:
  env = Env(SortLego(), BLIND)
  env.reset([0])
  port = _free_port()
  page = server.Page(
    env.model, env.sim.geom_xpos[0], env.sim.geom_xmat[0], FOCUS, port, None, None
  )
  html = urllib.request.urlopen(f"http://localhost:{port}/").read().decode()
  assert 'id="enter-vr"' in html
  with connect(f"ws://localhost:{port}/", max_size=None) as browser:
    staged = json.loads(browser.recv())
    assert (
      staged["type"] == "scene" and len(staged["geoms"]) == len(parts(env.model)) > 0
    )
    for buttons in ([False] * 6, [False, True, False, False, True, False]):
      hand = {"pose": [0.1, 1.0, -0.3, 1, 0, 0, 0], "trigger": 0.25}
      hand |= {"buttons": buttons}
      message = {"type": "input", "t": 1.2345, "head": [0, 1.6, 0, 1, 0, 0, 0]}
      browser.send(json.dumps(message | {"hands": {"right": hand}}))
    browser.send(json.dumps({"type": "command", "name": "recenter"}))
    poll = _polled(lambda poll: "recenter" in poll.commands, page)
    assert poll.commands == ["keep", "recenter"]
    assert poll.inputs is not None and poll.inputs.hands["right"].trigger == 0.25
    assert poll.inputs.drawn == 1.2345, "the headset's frame time, in seconds"
    anchor = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    page.show(env.sim.geom_xpos[0], env.sim.geom_xmat[0], anchor, {})
    frame = json.loads(browser.recv())
    assert frame["type"] == "frame" and frame["anchor"] == np.eye(4).ravel().tolist()
