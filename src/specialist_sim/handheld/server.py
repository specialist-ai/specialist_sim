"""The handheld page and its socket on one port: a console in a desktop
tab, the scene in AR or VR in the Quest's browser (``page.html``).

WebXR needs a secure context: ``localhost`` over ``adb reverse`` (``keep_reversed``) or a certificate.
Messages are JSON unless noted:

* page to server: ``input`` each XR frame (its time ``t`` in ms, head and tracked controllers,
  poses ``[x, y, z, w, x, y, z]`` in ``local-floor`` space, trigger, buttons, axes), ``command``
  (one of ``COMMANDS``), ``layout`` (the mode for the trays to come) and ``nudge`` (a step
  of the scene).
* server to page: ``scene`` once (``stage``), ``frame`` at ``RATE`` (moving geom poses as float32
  ``[x, y, z, w, x, y, z]``, the anchor, the status, the head and controllers in the room frame),
  and ``pulse``.
"""

from __future__ import annotations

import base64
import json
import queue
import shutil
import ssl
import subprocess
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.sync.server import ServerConnection, serve

from specialist_sim.arrays import Float64Array
from specialist_sim.render import parts

__all__ = [
  "HANDS",
  "PORT",
  "RATE",
  "Controller",
  "Inputs",
  "Nudge",
  "Page",
  "keep_reversed",
]

PAGE = Path(__file__).parent / "page.html"
HANDS = ("left", "right")
"""The controllers, by the hand that holds each, in ``SLOTS`` order."""
PORT = 8012
RATE = 60.0
"""Frames of the sim the page is sent a second."""
COMMANDS = ("record", "drop", "recenter")
"""What the demonstrator asks of a recording: start it, or keep it and start the next. Drop it and pause
on a new tray. Stand the scene in front of the headset again."""
BUTTONS = {("right", 4): "record", ("right", 5): "drop", ("left", 4): "recenter"}
"""What a Touch controller's buttons command, by their index in a WebXR ``xr-standard`` gamepad:
A records, B drops, X recenters."""
TRACKED = 0.5
"""Seconds since its last frame before the headset reads as gone."""
REVERSE_EVERY = 3.0
"""Seconds between checks of the adb reverse."""


@dataclass(frozen=True)
class Controller:
  pose: Float64Array
  """``(7,)`` in headset space."""
  trigger: float
  stick: tuple[float, float] = (0.0, 0.0)
  """The thumbstick, ``(x, y)`` as a gamepad reads it: x positive right, y negative pushed forward."""


@dataclass(frozen=True)
class Nudge:
  """A step the console moves the scene by: along the floor the way the demonstrator faces and to their
  right, up, meters, and a turn about the vertical through the tray, radians counterclockwise."""

  forward: float = 0.0
  right: float = 0.0
  up: float = 0.0
  turn: float = 0.0


@dataclass(frozen=True)
class Inputs:
  head: Float64Array
  """``(7,)`` in headset space."""
  hands: dict[str, Controller]
  """The controllers tracked, by hand."""
  at: float
  """When the headset sent it, ``time.perf_counter()``."""
  drawn: float | None = None
  """When the headset drew the frame, seconds on the headset's own clock: steadier
  than ``at``, which carries the link's jitter. ``None`` from a page that does not
  say."""


def _packed(array: np.ndarray[Any, Any]) -> str:
  return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


def _pose(xyz_wxyz: list[float]) -> Float64Array:
  pose = np.asarray(xyz_wxyz, np.float64)
  if pose.shape != (7,) or not np.all(np.isfinite(pose)):
    raise ValueError(f"a pose is 7 finite numbers, got {xyz_wxyz}")
  return pose


class Page:
  """The page and its socket, served from a thread of their own. ``model`` is what the page draws."""

  def __init__(
    self,
    model: mujoco.MjModel,
    port: int = PORT,
    cert: Path | None = None,
    key: Path | None = None,
  ):
    self.model = model
    self.drawn = parts(model)
    """What the tracer draws, which is what a page draws."""
    self.moving = [
      row for row, part in enumerate(self.drawn) if model.body_weldid[part.body] != 0
    ]
    self._lock = threading.Lock()
    self._inputs: Inputs | None = None
    self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()
    self._layouts: queue.SimpleQueue[str] = queue.SimpleQueue()
    self._nudges: queue.SimpleQueue[Nudge] = queue.SimpleQueue()
    self._pulses: list[queue.SimpleQueue[str]] = []
    self._scene = ""
    self._frame, self._version = "", 0
    secure = None
    if cert is not None:
      secure = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
      secure.load_cert_chain(cert, key)
    self._server = serve(
      self._talk, "0.0.0.0", port, process_request=_page, ssl=secure, max_size=2**20
    )
    threading.Thread(
      target=self._server.serve_forever, name="page", daemon=True
    ).start()

  def _texture(self, part: Any) -> dict[str, Any] | None:
    """A box's diffuse map as RGBA rows, top row first, or ``None``."""
    if part.texture is None:
      return None
    height, width, _ = part.texture.shape
    rgba = np.full((height, width, 4), 255, np.uint8)
    rgba[..., :3] = np.round(part.texture * 255.0).astype(np.uint8)
    return {"width": width, "height": height, "rgba": _packed(rgba)}

  def _poses(self, data: mujoco.MjData, rows: list[int]) -> np.ndarray[Any, Any]:
    out = np.empty((len(rows), 7), np.float32)
    quaternion = np.empty(4)
    for at, row in enumerate(rows):
      geom = self.drawn[row].geom
      mujoco.mju_mat2Quat(quaternion, data.geom_xmat[geom])
      out[at, :3], out[at, 3:] = data.geom_xpos[geom], quaternion
    return out

  def stage(
    self,
    data: mujoco.MjData,
    focus: Float64Array,
    panel: Float64Array,
    modes: tuple[str, ...],
    mode: str,
  ) -> None:
    """Build the scene message from ``data``, which places the geoms that never
    move: before a page connects. ``focus`` is the room point the desktop view
    looks at, ``panel`` where the status panel stands facing the demonstrator,
    ``modes`` the layouts the page offers and ``mode`` the one in use."""
    model, geoms = self.model, []
    for part in self.drawn:
      drawn: dict[str, Any] = {"color": part.albedo.tolist()}
      if model.geom_type[part.geom] == mujoco.mjtGeom.mjGEOM_MESH:
        drawn |= {
          "kind": "mesh",
          "vertices": _packed(part.vertices),
          "faces": _packed(part.faces.astype(np.uint32).ravel()),
        }
      else:
        drawn |= {
          "kind": "box",
          "size": part.size.tolist(),
          "texture": self._texture(part),
        }
      geoms.append(drawn)
    poses = self._poses(data, list(range(len(self.drawn))))
    message = {
      "type": "scene",
      "geoms": geoms,
      "moving": self.moving,
      "poses": _packed(poses),
      "panel": [float(value) for value in panel],
      "focus": [float(value) for value in focus],
      "hands": list(HANDS),
      "modes": list(modes),
      "mode": mode,
    }
    self._scene = json.dumps(message)

  def publish(
    self,
    data: mujoco.MjData,
    anchor: Float64Array,
    status: dict[str, Any],
    room: dict[str, Float64Array] | None = None,
  ) -> None:
    """What every page draws next: the moving geoms as ``data`` holds them, the anchor ``(4, 4)``
    that stands the room frame in headset space, ``status`` for the demonstrator, and ``room``, the head
    and the controllers in the room frame, ``{"head": (7,), "left": (7,), ...}``, for the console."""
    room = room or {}
    message = {
      "type": "frame",
      "poses": _packed(self._poses(data, self.moving)),
      "anchor": anchor.T.ravel().tolist(),  # column major, as three.js reads a matrix
      "status": status,
      "head": room["head"].tolist() if "head" in room else None,
      "controllers": {hand: room[hand].tolist() for hand in HANDS if hand in room},
    }
    with self._lock:
      self._frame = json.dumps(message)
      self._version += 1

  def pulse(self, hand: str) -> None:
    """A short buzz on ``hand``'s controller."""
    with self._lock:
      for pulses in self._pulses:
        pulses.put(hand)

  def latest(self) -> Inputs | None:
    """The newest frame the headset sent, ``None`` before the first or once it has gone quiet."""
    with self._lock:
      inputs = self._inputs
    return (
      inputs
      if inputs is not None and time.perf_counter() - inputs.at < TRACKED
      else None
    )

  def commands(self) -> list[str]:
    """Every command since the last call, from the controllers and the console, oldest first."""
    return _drained(self._commands)

  def layouts(self) -> list[str]:
    """Every mode the console asked the next tray be laid in, oldest first."""
    return _drained(self._layouts)

  def nudges(self) -> list[Nudge]:
    """Every step the console asked the scene moved by since the last call, oldest first."""
    return _drained(self._nudges)

  def _read(self, message: dict[str, Any], held: dict[str, list[bool]]) -> None:
    kind = message.get("type")
    if kind == "command" and message.get("name") in COMMANDS:
      self._commands.put(str(message["name"]))
    elif kind == "nudge":
      self._nudges.put(
        Nudge(
          *(
            float(message.get(name, 0.0)) for name in ("forward", "right", "up", "turn")
          )
        )
      )
    elif kind == "layout":
      self._layouts.put(str(message["mode"]))
    elif kind == "input":
      hands = {}
      for hand, state in message.get("hands", {}).items():
        buttons = [bool(pressed) for pressed in state.get("buttons", [])]
        before = held.get(hand, [])
        for index, pressed in enumerate(buttons):
          was = index < len(before) and before[index]
          if pressed and not was and (hand, index) in BUTTONS:
            self._commands.put(BUTTONS[(hand, index)])
        held[hand] = buttons
        stick = ([float(value) for value in state.get("axes", [])][2:4] + [0.0, 0.0])[
          :2
        ]  # xr-standard: 2, 3
        hands[hand] = Controller(
          _pose(state["pose"]),
          float(state.get("trigger", 0.0)),
          (stick[0], stick[1]),
        )
      drawn = message.get("t")
      inputs = Inputs(
        _pose(message["head"]),
        hands,
        time.perf_counter(),
        None if drawn is None else float(drawn) / 1000.0,  # the page's milliseconds
      )
      with self._lock:
        self._inputs = inputs

  def _talk(self, connection: ServerConnection) -> None:
    connection.send(self._scene)
    pulses: queue.SimpleQueue[str] = queue.SimpleQueue()
    with self._lock:
      self._pulses.append(pulses)
    shown = -1
    held: dict[str, list[bool]] = {}
    try:
      while True:
        try:
          self._read(json.loads(connection.recv(timeout=0.5 / RATE)), held)
        except TimeoutError:
          pass
        except (ValueError, KeyError, TypeError) as malformed:
          print(f"page: a malformed message ({malformed})")
        with self._lock:
          frame, version = self._frame, self._version
        if version != shown and frame:
          shown = version
          connection.send(frame)
        for hand in _drained(pulses):
          connection.send(json.dumps({"type": "pulse", "hand": hand}))
    except ConnectionClosed:
      return
    finally:
      with self._lock:
        self._pulses.remove(pulses)


def _drained[T](taken: queue.SimpleQueue[T]) -> list[T]:
  out: list[T] = []
  while True:
    try:
      out.append(taken.get_nowait())
    except queue.Empty:
      return out


def _page(connection: ServerConnection, request: Request) -> Response | None:
  """The page on a plain request, the socket on an upgrade."""
  if request.headers.get("Upgrade", "").lower() == "websocket":
    return None
  response = connection.respond(HTTPStatus.OK, PAGE.read_text())
  # Setting a header appends to a multidict, so the built-in type goes first.
  del response.headers["Content-Type"]
  response.headers["Content-Type"] = "text/html; charset=utf-8"
  response.headers["Cache-Control"] = "no-store"
  return response


def _reverse(port: int) -> str:
  """Make ``localhost:port`` on a headset on USB reach this machine (``adb reverse``), and say how it
  stands. A reconnect drops the reverse, and without it the page cannot load in the headset."""
  adb = shutil.which("adb")
  if adb is None:
    return "no adb"
  try:
    devices = subprocess.run(
      [adb, "devices"], capture_output=True, text=True, timeout=5
    ).stdout
    if not any(line.endswith("\tdevice") for line in devices.splitlines()[1:]):
      return "no headset on USB"
    listed = subprocess.run(
      [adb, "reverse", "--list"], capture_output=True, text=True, timeout=5
    ).stdout
    if f"tcp:{port} tcp:{port}" not in listed:
      subprocess.run(
        [adb, "reverse", f"tcp:{port}", f"tcp:{port}"],
        capture_output=True,
        timeout=5,
        check=True,
      )
  except (subprocess.SubprocessError, OSError):
    return "adb did not answer"
  return "reversed"


def keep_reversed(port: int) -> None:
  """Keep the adb reverse in place from a thread, printing each change: adb takes tens of
  milliseconds to answer, which the sim's loop cannot spare."""

  def check() -> None:
    standing = None
    while True:
      if (state := _reverse(port)) != standing:
        standing = state
        print(f"headset over USB: {state}")
      time.sleep(REVERSE_EVERY)

  threading.Thread(target=check, name="adb-reverse", daemon=True).start()
