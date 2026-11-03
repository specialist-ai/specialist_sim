"""The LEGO setup as MuJoCo specs, in the room frame of
:mod:`specialist_sim.sort_lego.scene`.

* :func:`fixtures`: the desk, the TILLGANG tray and the two bins, all static.
* :func:`brick`: one 2 x 4 brick on a free joint. It is drawn with studs but
  collides as one box.
* :func:`spec`: the fixtures and every brick, parked in its bin, which the rig
  stands the hands in.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from specialist_sim import desk
from specialist_sim.sort_lego import scene

__all__ = ["ASSETS", "COLOR_RGB", "brick", "fixtures", "spec"]

ASSETS = Path(__file__).parent / "assets"
RED = (0.74, 0.11, 0.09)
BLUE = (0.03, 0.33, 0.75)
COLOR_RGB = {"red": RED, "blue": BLUE}
BIN_RGB = {"red": (0.74, 0.16, 0.17), "blue": (0.06, 0.19, 0.75)}
"""The AkroBins' plastic: each bin's hue in a photo of it beside its bricks, at its bricks'
brightness here. The blue bin is a purer blue than the bricks."""
BRICK_FRICTION = (0.8, 0.005, 0.0001)
TRAY_FRICTION = (0.5, 0.005, 0.0001)
"""The tray's plastic, floor and rim alike. MuJoCo takes the larger of two
geoms' frictions, so on the tray a brick slides at its own 0.8: under tan 45 =
1, so it slides down the rim's slope rather than resting on it."""
THICK = 0.01
"""Half the thickness every static collider a brick can be tossed at is grown
to, meters."""


def _box(
  body: mujoco.MjsBody,
  name: str,
  pos: Sequence[float],
  size: Sequence[float],
  rgba: Sequence[float] | None = None,
  group: int = 3,
  **kwargs: Any,
) -> None:
  body.add_geom(
    name=name,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=list(pos),
    size=list(size),
    rgba=list(rgba) if rgba is not None else [0.5, 0.5, 0.5, 1.0],
    group=group,
    **kwargs,
  )


def _container(
  spec: mujoco.MjSpec, body: mujoco.MjsBody, color: str, card: str
) -> None:
  """One clear bin, its long side along x and the card of color ``card`` on the face
  at +x, toward the near side."""
  length, width, height = scene.BINS["clear"].size
  half = scene.CLEAR_WALL / 2.0
  radius, arc = 0.020, 4
  corners = ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
  outline = []
  for turn, (sx, sy) in enumerate(corners):
    for k in range(arc + 1):
      angle = np.pi / 2.0 * (turn + k / arc)
      outline.append(
        (
          sx * (length / 2.0 - radius) + radius * np.cos(angle),
          sy * (width / 2.0 - radius) + radius * np.sin(angle),
        )
      )
  count = len(outline)
  vertices = [(x, y, 2.0 * half) for x, y in outline]
  vertices += [(x, y, height) for x, y in outline] + [(0.0, 0.0, 2.0 * half)]
  faces = []
  for a in range(count):
    b = (a + 1) % count
    faces += [(a, b, count + b), (a, count + b, count + a), (2 * count, b, a)]
  faces += [face[::-1] for face in faces]
  mesh = spec.add_mesh(name=f"{color}_bin_shell")
  mesh.uservert = np.ravel(vertices).tolist()
  mesh.userface = np.ravel(faces).tolist()
  body.add_geom(
    name=f"{color}_bin_shell",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=f"{color}_bin_shell",
    rgba=[0.84, 0.85, 0.86, 1.0],
    contype=0,
    conaffinity=0,
    group=2,
  )
  # A tossed brick falls a few millimeters a step, more than a 2 mm wall is thick, so every
  # collider is grown away from the inside: the floor down into the desk, the walls outward.
  # The inside keeps its measured size. Static geoms never collide with one another.
  t = THICK
  walls = {
    "floor": ((0.0, 0.0, 2 * half - t), (length / 2.0, width / 2.0, t)),
    "wall_far": (
      (2 * half - length / 2.0 - t, 0.0, height / 2.0),
      (t, width / 2.0 + t, height / 2.0),
    ),
    "wall_near": (
      (length / 2.0 - 2 * half + t, 0.0, height / 2.0),
      (t, width / 2.0 + t, height / 2.0),
    ),
    "wall_left": (
      (0.0, width / 2.0 - 2 * half + t, height / 2.0),
      (length / 2.0, t, height / 2.0),
    ),
    "wall_right": (
      (0.0, 2 * half - width / 2.0 - t, height / 2.0),
      (length / 2.0, t, height / 2.0),
    ),
  }
  for part, (pos, size) in walls.items():
    _box(body, f"{color}_bin_{part}", pos, size, rgba=(0.84, 0.85, 0.86, 0.3))
  spec.add_texture(
    name=f"{color}_label",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    file=str(ASSETS / f"label_{card}.png"),
  )
  material = spec.add_material(name=f"{color}_label")
  material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"{color}_label"
  body.add_geom(
    name=f"{color}_bin_card",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=[length / 2.0 + 0.00025, 0.0, 0.047],
    quat=[0.5, 0.5, 0.5, 0.5],
    size=[0.063 / 2.0, 0.041 / 2.0, 0.00025],
    material=f"{color}_label",
    contype=0,
    conaffinity=0,
    group=2,
  )


def _akrobin_meshes(spec: mujoco.MjSpec) -> None:
  """The AkroBin's shell, drawn from its CAD, and its two side walls as convex prisms, once
  for both bins.

  Each side wall spans the inner face to the outer flange, 8.7 mm, and in profile runs from
  the back's outer face along the top of the wall, down its sloped front to the label lip,
  and back along the desk. MuJoCo collides the hull of the profile's points, which bridges
  the sloped front up to 3 mm above the real edge.
  """
  shell = spec.add_mesh(name="akrobin", file=str(ASSETS / "akrobin_30210.stl"))
  shell.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
  profile = np.array(
    [
      (-0.0689, 0.0),
      (-0.0689, 0.0717),
      (0.0446, 0.0717),
      (0.0516, 0.0681),
      (0.0566, 0.0561),
      (0.0606, 0.0464),
      (0.0646, 0.0380),
      (0.0660, 0.0380),
      (0.0660, 0.0),
    ]
  )
  inner, outer = scene.BINS["akrobins"].inside[2], scene.BINS["akrobins"].size[1] / 2
  for side, sign in (("left", 1.0), ("right", -1.0)):
    wall = spec.add_mesh(name=f"akrobin_wall_{side}")
    wall.uservert = np.ravel(
      [(x, sign * across, z) for across in (inner, outer) for x, z in profile]
    ).tolist()


def _akrobin(body: mujoco.MjsBody, color: str, drawn: str) -> None:
  """One AkroBin, open front toward the near side at +x, its plastic in ``drawn``'s color.

  The colliders sit on the CAD's inner surfaces. The floor and the 45 degree ramp are
  ``2 * THICK`` thick below them, as every static collider a brick can be tossed at is; the
  walls are their real thickness, 7 mm and more, and the label lip its real 3.4 mm.
  """
  bins = scene.BINS["akrobins"]
  back, front, top = bins.floor
  half_width = bins.inside[2]
  body.add_geom(
    name=f"{color}_bin_shell",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="akrobin",
    rgba=[*BIN_RGB[drawn], 1.0],
    contype=0,
    conaffinity=0,
    group=2,
  )
  # The same plastic as the tray: a brick slides down the ramp, tan 45 = 1, at its own 0.8.
  collider = (*BIN_RGB[drawn], 0.3)
  _box(
    body,
    f"{color}_bin_floor",
    ((back + front) / 2, 0.0, top - THICK),
    ((front - back) / 2 + 0.005, half_width, THICK),
    rgba=collider,
    friction=TRAY_FRICTION,
  )
  # The ramp's face rises at 45 degrees from the floor's front edge, z = x - (front - top),
  # to the label lip's back face.
  ramp = front - top
  foot, head = front - 0.005, bins.inside[1] - 0.0034
  normal = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
  middle = np.array([(foot + head) / 2, 0.0, (foot + head) / 2 - ramp]) - THICK * normal
  _box(
    body,
    f"{color}_bin_ramp",
    middle,
    ((head - foot) / 2 * np.sqrt(2.0) + 0.002, half_width, THICK),
    quat=[np.cos(np.pi / 8), 0.0, -np.sin(np.pi / 8), 0.0],
    rgba=collider,
    friction=TRAY_FRICTION,
  )
  _box(
    body,
    f"{color}_bin_lip",
    (bins.inside[1] - 0.0017, 0.0, (0.011 + 0.038) / 2),
    (0.0017, bins.size[1] / 2, (0.038 - 0.011) / 2),
    rgba=collider,
    friction=TRAY_FRICTION,
  )
  _box(
    body,
    f"{color}_bin_back",
    ((back - bins.size[0] / 2) / 2, 0.0, bins.size[2] / 2),
    ((back + bins.size[0] / 2) / 2, bins.size[1] / 2, bins.size[2] / 2),
    rgba=collider,
    friction=TRAY_FRICTION,
  )
  for side in ("left", "right"):
    body.add_geom(
      name=f"{color}_bin_wall_{side}",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=f"akrobin_wall_{side}",
      group=3,
      rgba=collider,
      friction=TRAY_FRICTION,
    )


def _tray(root: mujoco.MjsBody) -> None:
  """The tray, its long side along y: a flat floor and a rim whose inner wall slopes up from
  the floor at ``TRAY_WALL_ANGLE`` to a flat top ``TRAY_LIP`` above the desk.

  Each side of the rim is two boxes: its flat top, and a box tilted to the wall's angle whose
  upper face is the slope, from the top's inner edge down into the floor. Like every static
  collider a brick can be tossed at they are ``2 * THICK`` thick on the solid side, and each
  runs the side's full length, so at the corners they overlap into a mitered basin.
  """
  tray = root.add_body(name="tray", pos=[*scene.TRAY_CENTER, scene.DESK_TOP])
  gray = (0.24, 0.24, 0.25, 1.0)
  half = np.array(scene.TRAY_SIZE) / 2
  rim, lip, floor = scene.TRAY_RIM, scene.TRAY_LIP, scene.TRAY_FLOOR_THICKNESS
  angle = scene.TRAY_WALL_ANGLE
  run = (lip - floor) / np.tan(angle)
  top = rim - run  # the flat top's width
  _box(
    tray,
    "tray_floor",
    (0, 0, floor - THICK),
    (half[0], half[1], THICK),
    gray,
    group=2,
    friction=TRAY_FRICTION,
  )
  up = np.array([0.0, 0.0, 1.0])
  # The flat tops in a body of their own, so a contact sensor can tell the rim's top from the
  # sloped walls a brick lying against them is taken from.
  rim_body = tray.add_body(name="tray_rim")
  for side, axis, sign in (
    ("near", 0, 1),
    ("far", 0, -1),
    ("left", 1, 1),
    ("right", 1, -1),
  ):
    out = np.zeros(3)
    out[axis] = sign  # the side's outward normal
    along = np.cross(up, out)
    length = half[1 - axis]
    edge = half[axis]
    if top > 0:
      center = out * (edge - top / 2) + up * (lip - THICK)
      size = np.abs(out) * top / 2 + np.abs(along) * length + up * THICK
      _box(rim_body, f"tray_top_{side}", center, size, gray, group=2)
    # The slope: down and in from the top's inner edge, carried a further THICK below the
    # floor so the two meet without a seam. Its normal points up and into the tray.
    down = -out * np.cos(angle) - up * np.sin(angle)
    normal = up * np.cos(angle) - out * np.sin(angle)
    start = out * (edge - top) + up * lip
    span = run / np.cos(angle) + THICK
    center = start + down * span / 2 - normal * THICK
    frame = np.stack([down, np.cross(normal, down), normal], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, frame.flatten())
    _box(
      tray,
      f"tray_slope_{side}",
      center,
      (span / 2, length, THICK),
      gray,
      group=2,
      quat=list(quat),
      friction=TRAY_FRICTION,
    )


def fixtures(bins: str, swapped: bool = False) -> mujoco.MjSpec:
  """The desk, the tray and the two bins of kind ``bins``, standing where the
  takes put them, each bin wearing the other's color when ``swapped``: a clear
  bin its card, an AkroBin its plastic."""
  spec = mujoco.MjSpec()
  spec.compiler.texturedir = str(ASSETS)
  root = spec.worldbody.add_body(name="fixtures")
  desk.stand(root)
  _tray(root)
  if bins == "akrobins":
    _akrobin_meshes(spec)
  for color in scene.COLORS:
    body = root.add_body(
      name=f"{color}_bin", pos=[*scene.BINS[bins].middle(color), scene.DESK_TOP]
    )
    drawn = scene.COLORS[1 - scene.COLORS.index(color)] if swapped else color
    if bins == "akrobins":
      _akrobin(body, color, drawn)
    else:
      _container(spec, body, color, drawn)
  return spec


def brick(name: str, color: str, at: tuple[float, float, float]) -> mujoco.MjSpec:
  """One brick on a free joint, lying flat with its long side along x, centered on its
  collision box (studs included), so its mass is too. ``at`` is where it stands in the
  model's own reference pose, which is where the solver sizes its contact buffers."""
  length, width, height = scene.BRICK
  rise = 0.00185
  body_height = height - rise
  stud, pitch, wall, ceiling = 0.0049 / 2, 0.008, 0.0012, 0.001
  sides = 12
  bottom = -height / 2.0
  top, inside = bottom + body_height, bottom + body_height - ceiling
  vertices: list[tuple[float, float, float]] = []
  faces: list[tuple[int, int, int]] = []

  def points(*corners: tuple[float, float, float]) -> list[int]:
    vertices.extend(corners)
    return list(range(len(vertices) - len(corners), len(vertices)))

  def rectangle(z: float, inset: float) -> list[int]:
    x, y = length / 2.0 - inset, width / 2.0 - inset
    return points((-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z))

  def quad(a: int, b: int, c: int, d: int) -> None:
    faces.extend([(a, b, c), (a, c, d)])

  def band(lower: list[int], upper: list[int], outward: bool) -> None:
    for k in range(len(lower)):
      j = (k + 1) % len(lower)
      if outward:
        quad(lower[k], lower[j], upper[j], upper[k])
      else:
        quad(lower[k], upper[k], upper[j], lower[j])

  def rim(outer: list[int], inner: list[int]) -> None:
    for k in range(len(outer)):
      j = (k + 1) % len(outer)
      quad(outer[k], inner[k], inner[j], outer[j])

  outer_low, outer_high = rectangle(bottom, 0.0), rectangle(top, 0.0)
  inner_low, inner_high = rectangle(bottom, wall), rectangle(inside, wall)
  band(outer_low, outer_high, True)
  band(inner_low, inner_high, False)
  quad(*outer_high)
  quad(*reversed(inner_high))
  rim(outer_low, inner_low)
  for x in (-1.5 * pitch, -0.5 * pitch, 0.5 * pitch, 1.5 * pitch):
    for y in (-0.5 * pitch, 0.5 * pitch):
      turns = 2.0 * np.pi * np.arange(sides) / sides
      base = points(*[(x + stud * np.cos(a), y + stud * np.sin(a), top) for a in turns])
      cap = points(
        *[(x + stud * np.cos(a), y + stud * np.sin(a), top + rise) for a in turns]
      )
      band(base, cap, True)
      (middle,) = points((x, y, top + rise))
      faces.extend((middle, cap[k], cap[(k + 1) % sides]) for k in range(sides))

  spec = mujoco.MjSpec()
  mesh = spec.add_mesh(name=f"{name}_mesh")
  mesh.uservert = np.ravel(vertices).tolist()
  mesh.userface = np.ravel(faces).tolist()
  mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
  body = spec.worldbody.add_body(name=name, pos=list(at))
  body.add_joint(name=f"{name}_joint", type=mujoco.mjtJoint.mjJNT_FREE)
  body.add_geom(
    name=f"{name}_visual",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=f"{name}_mesh",
    rgba=[*COLOR_RGB[color], 1.0],
    contype=0,
    conaffinity=0,
    density=0,
    group=2,
  )
  body.add_geom(
    name=f"{name}_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(length / 2.0, width / 2.0, height / 2.0),
    mass=scene.BRICK_MASS,
    group=3,
    friction=list(BRICK_FRICTION),
    rgba=[*COLOR_RGB[color], 1.0],
  )
  return spec


def spec(colors: tuple[str, ...], bins: str, swapped: bool = False) -> mujoco.MjSpec:
  """The fixtures and a brick of each of ``colors``, parked in its own bin.
  ``swapped`` draws every brick and bin in the other color. Physics and
  scoring keep each brick's own, so the brick drawn blue belongs in the bin
  drawn blue."""
  other = dict(zip(scene.COLORS, scene.COLORS[::-1], strict=True))
  room = fixtures(bins, swapped)
  count = dict.fromkeys(scene.COLORS, 0)
  for index, color in enumerate(colors):
    at = scene.BINS[bins].parked(color, count[color])
    count[color] += 1
    drawn = other[color] if swapped else color
    room.attach(
      brick(f"brick_{index}", drawn, at), prefix="", frame=room.worldbody.add_frame()
    )
  return room
