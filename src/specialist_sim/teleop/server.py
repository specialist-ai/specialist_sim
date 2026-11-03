"""The teleoperation page and its socket on one port: a console in a desktop
tab, the scene in AR or VR in the Quest's browser (``page.html``).

WebXR needs a secure context: ``localhost`` over ``adb reverse`` (``keep_reversed``) or a certificate.
Messages are JSON unless noted:

* page to server: ``input`` each XR frame (head and tracked controllers, poses ``[x, y, z, w, x, y,
  z]`` in ``local-floor`` space, trigger, buttons, axes), ``command`` (one of ``COMMANDS``),
  ``layout`` (mode and bricks for the trays to come) and ``nudge`` (a step of the scene).
* server to page: ``scene`` once (``stage``), ``frame`` at ``RATE`` (moving geom poses as float32
  ``[x, y, z, w, x, y, z]``, the anchor, the status, the head and controllers in the room frame),
  ``pulse``, and binary lens pictures: one byte of name length, the name, then a JPEG.
"""

from __future__ import annotations

import base64
import io
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

from specialist_sim.arrays import Float64Array, UInt8Array

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
DRAWN = (1, 2)
"""The geom groups the scene draws with, as the lens does."""
COMMANDS = ("record", "drop", "recenter")
"""What the operator asks of a recording: start it, or keep it and start the next. Drop it and pause
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
  """A step the console moves the scene by: along the floor the way the operator faces and to their
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
    self.drawn = [
      geom
      for geom in range(model.ngeom)
      if model.geom_group[geom] in DRAWN
      # ``==``, not ``in``: a numpy integer is never found in a tuple of MuJoCo's enums.
      and (
        model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
        or model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
      )
      and self._color(geom)[3] >= 0.5
    ]
    self.moving = [
      row
      for row, geom in enumerate(self.drawn)
      if model.body_weldid[model.geom_bodyid[geom]] != 0
    ]
    self._lock = threading.Lock()
    self._inputs: Inputs | None = None
    self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()
    self._layouts: queue.SimpleQueue[dict[str, Any]] = queue.SimpleQueue()
    self._nudges: queue.SimpleQueue[Nudge] = queue.SimpleQueue()
    self._pulses: list[queue.SimpleQueue[str]] = []
    self._scene = ""
    self._frame, self._version = "", 0
    self._images: dict[str, bytes] = {}
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

  def _color(self, geom: int) -> Float64Array:
    material = self.model.geom_matid[geom]
    out: Float64Array = (
      self.model.mat_rgba[material] if material >= 0 else self.model.geom_rgba[geom]
    )
    return out

  def _texture(self, geom: int) -> dict[str, Any] | None:
    """A box's diffuse map as RGBA rows, top row first, or ``None``."""
    model, material = self.model, self.model.geom_matid[geom]
    texture = (
      -1
      if material < 0
      else int(model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB])
    )
    if texture < 0:
      return None
    width, height, channels = (
      int(v)
      for v in (
        model.tex_width[texture],
        model.tex_height[texture],
        model.tex_nchannel[texture],
      )
    )
    start = int(model.tex_adr[texture])
    pixels = model.tex_data[start : start + width * height * channels].reshape(
      height, width, channels
    )
    rgba = np.full((height, width, 4), 255, np.uint8)
    rgba[..., : min(channels, 4)] = pixels[..., :4]
    return {"width": width, "height": height, "rgba": _packed(rgba)}

  def _poses(self, data: mujoco.MjData, rows: list[int]) -> np.ndarray[Any, Any]:
    out = np.empty((len(rows), 7), np.float32)
    quaternion = np.empty(4)
    for at, row in enumerate(rows):
      geom = self.drawn[row]
      mujoco.mju_mat2Quat(quaternion, data.geom_xmat[geom])
      out[at, :3], out[at, 3:] = data.geom_xpos[geom], quaternion
    return out

  def stage(
    self,
    data: mujoco.MjData,
    floor: float,
    focus: Float64Array,
    panel: Float64Array,
    modes: tuple[str, ...],
    layout: dict[str, Any],
  ) -> None:
    """Build the scene message from ``data``, which places the geoms that never
    move: before a page connects. ``floor`` is the real floor's height in the
    room frame, ``focus`` the room point the desktop view looks at, ``panel``
    where the status panel stands facing the operator (the lens panels stand
    either side of it), ``modes`` the layouts the page offers and ``layout`` the
    one in use, ``{"mode", "bricks"}``."""
    model, geoms = self.model, []
    for geom in self.drawn:
      drawn: dict[str, Any] = {"color": self._color(geom).tolist()}
      if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        at, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        face, faces = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        drawn |= {
          "kind": "mesh",
          "vertices": _packed(model.mesh_vert[at : at + count].astype(np.float32)),
          "faces": _packed(model.mesh_face[face : face + faces].astype(np.uint32)),
        }
      else:
        drawn |= {
          "kind": "box",
          "size": model.geom_size[geom].tolist(),
          "texture": self._texture(geom),
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
      "floor": floor,
      "hands": list(HANDS),
      "modes": list(modes),
      "layout": layout,
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
    that stands the room frame in headset space, ``status`` for the operator, and ``room``, the head
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

  def show(self, images: dict[str, UInt8Array]) -> None:
    """Each lens's newest picture, ``(size, size, 3)`` uint8 by hand, for every page to show."""
    from PIL import Image

    encoded = {}
    for name, image in images.items():
      buffer = io.BytesIO()
      Image.fromarray(np.ascontiguousarray(image)).save(
        buffer, format="JPEG", quality=80
      )
      named = name.encode()
      encoded[name] = bytes([len(named)]) + named + buffer.getvalue()
    with self._lock:
      self._images = encoded

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

  def layouts(self) -> list[dict[str, Any]]:
    """Every layout the console asked for since the last call, ``{"mode", "bricks"}``, oldest first."""
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
      bricks = message.get("bricks")
      self._layouts.put(
        {
          "mode": str(message["mode"]),
          "bricks": None if bricks is None else int(bricks),
        }
      )
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
      inputs = Inputs(_pose(message["head"]), hands, time.perf_counter())
      with self._lock:
        self._inputs = inputs

  def _talk(self, connection: ServerConnection) -> None:
    connection.send(self._scene)
    pulses: queue.SimpleQueue[str] = queue.SimpleQueue()
    with self._lock:
      self._pulses.append(pulses)
    shown = -1
    sent: dict[str, bytes] = {}
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
          frame, version, images = self._frame, self._version, self._images
        if version != shown and frame:
          shown = version
          connection.send(frame)
        for name, image in images.items():
          if sent.get(name) is not image:
            sent[name] = image
            connection.send(image)
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
