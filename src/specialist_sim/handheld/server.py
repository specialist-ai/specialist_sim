"""The handheld page (``page.html``) and its socket on one port: the scene
in AR or VR in the Quest's browser, and around an orbiting camera in a desktop tab.

WebXR needs a secure context: ``localhost`` over ``adb reverse`` or a certificate.
Messages are JSON, poses ``[x, y, z, w, x, y, z]``:

* page to server: ``input`` each XR frame (time ``t``, head and tracked controllers
  in ``local-floor`` space, trigger, buttons) and ``command`` (``recenter``,
  as a session starts).
* server to page: ``scene`` once a page connects (every drawn geom, its pose, the
  focus), ``frame`` at each ``show`` (moving geom poses as base64
  float32, the anchor and the status).
"""

import base64
import json
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

from specialist_sim import rotation
from specialist_sim.arrays import Float32Array, Float64Array, UInt32Array
from specialist_sim.camera.render import parts
from specialist_sim.handheld.xr import Controller, Inputs

# What a Touch controller's buttons command, by their index in a WebXR
# xr-standard gamepad: the left starts, the right ends. X records, Y recenters,
# A keeps, B drops.
BUTTONS = {
  ("left", 4): "record",
  ("left", 5): "recenter",
  ("right", 4): "keep",
  ("right", 5): "drop",
}


@dataclass(frozen=True)
class Poll:
  """What the page asked since the last poll."""

  inputs: Inputs | None
  """The headset's newest frame, ``None`` before the first or once it goes quiet."""
  commands: list[str]
  """From the controllers and the page, oldest first."""


def _packed(array: Float32Array | UInt32Array) -> str:
  return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


class Page:
  """The page and its socket, served from a thread of their own. ``model`` is
  what the page draws, ``geom_xpos`` ``(ngeom, 3)`` and ``geom_xmat`` ``(ngeom,
  9)`` where it stands, and ``focus`` the room point the desktop view looks at."""

  def __init__(
    self,
    model: mujoco.MjModel,
    geom_xpos: Float64Array,
    geom_xmat: Float64Array,
    focus: Float64Array,
    port: int,
    cert: Path | None,
    key: Path | None,
  ):
    self._drawn = parts(model)
    self._moving = [
      row for row, part in enumerate(self._drawn) if model.body_weldid[part.body] != 0
    ]
    geoms: list[dict[str, Any]] = []
    for part in self._drawn:
      if model.geom_type[part.geom] == mujoco.mjtGeom.mjGEOM_MESH:
        geoms.append(
          {
            "kind": "mesh",
            "vertices": _packed(part.vertices),
            "faces": _packed(part.faces.astype(np.uint32).ravel()),
          }
        )
      else:
        geoms.append({"kind": "box", "size": part.size.tolist()})
      geoms[-1]["color"] = part.albedo.tolist()
    self._scene = json.dumps(
      {
        "type": "scene",
        "geoms": geoms,
        "moving": self._moving,
        "poses": _packed(self._poses(geom_xpos, geom_xmat, range(len(self._drawn)))),
        "focus": focus.tolist(),
      }
    )
    self._lock = threading.Lock()
    self._inputs: Inputs | None = None
    self._commands: list[str] = []
    self._frame = ""
    self._version = 0
    secure = None
    if cert is not None:
      secure = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
      secure.load_cert_chain(cert, key)
    server = serve(self._talk, "0.0.0.0", port, process_request=_page, ssl=secure)
    threading.Thread(target=server.serve_forever, name="page", daemon=True).start()

  def _poses(
    self, geom_xpos: Float64Array, geom_xmat: Float64Array, rows: range | list[int]
  ) -> Float32Array:
    out = np.empty((len(rows), 7), np.float32)
    for at, row in enumerate(rows):
      geom = self._drawn[row].geom
      out[at, :3] = geom_xpos[geom]
      out[at, 3:] = rotation.from_matrix(geom_xmat[geom])
    return out

  def poll(self) -> Poll:
    with self._lock:
      inputs = self._inputs
      # The headset reads as gone 0.5 s after its last frame.
      fresh = inputs is not None and time.perf_counter() - inputs.at < 0.5
      poll = Poll(inputs if fresh else None, self._commands)
      self._commands = []
    return poll

  def show(
    self,
    geom_xpos: Float64Array,
    geom_xmat: Float64Array,
    anchor: Float64Array,
    status: dict[str, Any],
  ) -> None:
    """What every page draws next: the moving geoms where ``geom_xpos`` and
    ``geom_xmat`` stand them, the anchor ``(7,)`` that stands the room frame in
    headset space, and ``status`` for the demonstrator."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.to_matrix(anchor[3:])
    matrix[:3, 3] = anchor[:3]
    message = {
      "type": "frame",
      "poses": _packed(self._poses(geom_xpos, geom_xmat, self._moving)),
      "anchor": matrix.T.ravel().tolist(),  # column major, as three.js reads a matrix
      "status": status,
    }
    with self._lock:
      self._frame = json.dumps(message)
      self._version += 1

  def _read(self, message: dict[str, Any], held: dict[str, list[bool]]) -> None:
    kind = message["type"]
    with self._lock:
      if kind == "command":
        self._commands.append(str(message["name"]))
      elif kind == "input":
        hands = {}
        for hand, state in message["hands"].items():
          buttons = [bool(pressed) for pressed in state["buttons"]]
          before = held.get(hand, [])
          held[hand] = buttons
          self._commands += [
            BUTTONS[(hand, index)]
            for index, pressed in enumerate(buttons)
            if pressed
            and not (index < len(before) and before[index])
            and (hand, index) in BUTTONS
          ]
          hands[hand] = Controller(
            np.asarray(state["pose"], np.float64), float(state["trigger"])
          )
        self._inputs = Inputs(
          np.asarray(message["head"], np.float64),
          hands,
          time.perf_counter(),
          float(message["t"]),
        )

  def _talk(self, connection: ServerConnection) -> None:
    connection.send(self._scene)
    shown = -1
    held: dict[str, list[bool]] = {}
    try:
      while True:
        try:
          # Back within 5 ms, so a new frame leaves soon after it is shown.
          self._read(json.loads(connection.recv(timeout=0.005)), held)
        except TimeoutError:
          pass
        with self._lock:
          frame = self._frame
          version = self._version
        if version != shown and frame:
          shown = version
          connection.send(frame)
    except ConnectionClosed:
      return


def _page(connection: ServerConnection, request: Request) -> Response | None:
  """The page on a plain request, the socket on an upgrade."""
  if request.headers.get("Upgrade", "").lower() == "websocket":
    return None
  page = (Path(__file__).parent / "page.html").read_text()
  response = connection.respond(HTTPStatus.OK, page)
  # Setting a header appends to a multidict, so the built-in type goes first.
  del response.headers["Content-Type"]
  response.headers["Content-Type"] = "text/html; charset=utf-8"
  response.headers["Cache-Control"] = "no-store"
  return response


def _adb(*args: str) -> str:
  return subprocess.run(
    ["adb", *args], capture_output=True, text=True, timeout=5, check=True
  ).stdout


def _reverse(port: int) -> str:
  """Make ``localhost:port`` on a headset on USB reach this machine, and say how
  it stands. A USB reconnect drops the reverse."""
  try:
    if not any(line.endswith("\tdevice") for line in _adb("devices").splitlines()[1:]):
      return "no headset on USB"
    if f"tcp:{port} tcp:{port}" not in _adb("reverse", "--list"):
      _adb("reverse", f"tcp:{port}", f"tcp:{port}")
  except FileNotFoundError:
    return "no adb"
  except subprocess.SubprocessError, OSError:
    return "adb did not answer"
  return "reversed"


def keep_reversed(port: int) -> None:
  """Keep the adb reverse in place from a thread, printing each change."""

  def check() -> None:
    standing = None
    while True:
      if (state := _reverse(port)) != standing:
        standing = state
        print(f"headset over USB: {state}")
      time.sleep(3.0)

  threading.Thread(target=check, name="adb-reverse", daemon=True).start()
