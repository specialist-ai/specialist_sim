"""The LEGO setup of :mod:`specialist_sim.tasks.sort_lego.measures` as MuJoCo specs."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from specialist_sim import rotation
from specialist_sim.gripper import DRAWN
from specialist_sim.tasks import desk
from specialist_sim.tasks.sort_lego import measures

BRICK_RGB = {"red": (0.74, 0.11, 0.09), "blue": (0.03, 0.33, 0.75)}
BIN_RGB = {"red": (0.74, 0.16, 0.17), "blue": (0.06, 0.19, 0.75)}
BRICK_FRICTION = (0.8, 0.005, 0.0001)
# The tray's and the bins' plastic; MuJoCo takes the larger friction of two geoms.
TRAY_FRICTION = (0.5, 0.005, 0.0001)
# Half the thickness of the tray's boxes and the bins' floors and ramps.
HALF_THICK = 0.01


def _box(
  body: mujoco.MjsBody,
  name: str,
  pos: Sequence[float],
  size: Sequence[float],
  group: int = 3,
  **kwargs: Any,
) -> None:
  body.add_geom(
    name=name,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=list(pos),
    size=list(size),
    group=group,
    **kwargs,
  )


def _akrobin_meshes(spec: mujoco.MjSpec) -> None:
  """The AkroBin's shell from its CAD and its two side walls, shared by both bins.

  A wall spans the inner face to the outer flange. In profile it runs from the
  back's outer face over the wall top, down the sloped front to the label lip and
  back along the desk; MuJoCo collides the convex hull of these points.
  """
  shell = spec.add_mesh(
    name="akrobin", file=str(Path(__file__).parent / "assets" / "akrobin_30210.stl")
  )
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
  inner = measures.BIN_INSIDE[2]
  outer = measures.BIN_SIZE[1] / 2
  # Named by the operator's side: the bin's own +y is the operator's right.
  for side, sign in (("left", -1.0), ("right", 1.0)):
    wall = spec.add_mesh(name=f"akrobin_wall_{side}")
    wall.uservert = np.ravel(
      [(x, sign * across, z) for across in (inner, outer) for x, z in profile]
    ).tolist()


def _akrobin(body: mujoco.MjsBody, color: str) -> None:
  """One AkroBin in its color's plastic. The colliders sit on the CAD's inner
  surfaces; the floor and ramp are ``2 * HALF_THICK`` thick below them, the walls
  and the label lip their real thickness."""
  back, front, top = measures.BIN_FLOOR
  inside = measures.BIN_INSIDE
  size = measures.BIN_SIZE
  half_width = inside[2]
  body.add_geom(
    name=f"{color}_bin_shell",
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname="akrobin",
    rgba=[*BIN_RGB[color], 1.0],
    contype=0,
    conaffinity=0,
    group=DRAWN,
  )
  _box(
    body,
    f"{color}_bin_floor",
    ((back + front) / 2, 0.0, top - HALF_THICK),
    ((front - back) / 2 + 0.005, half_width, HALF_THICK),
    friction=TRAY_FRICTION,
  )
  # The ramp's face, z = x - (front - top), rises from the floor's front edge to the
  # label lip's back face.
  ramp = front - top
  foot = front - 0.005
  head = inside[1] - 0.0034
  normal = np.array([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
  middle = (
    np.array([(foot + head) / 2, 0.0, (foot + head) / 2 - ramp]) - HALF_THICK * normal
  )
  _box(
    body,
    f"{color}_bin_ramp",
    middle,
    ((head - foot) / 2 * np.sqrt(2.0) + 0.002, half_width, HALF_THICK),
    quat=[np.cos(np.pi / 8), 0.0, -np.sin(np.pi / 8), 0.0],
    friction=TRAY_FRICTION,
  )
  _box(
    body,
    f"{color}_bin_lip",
    (inside[1] - 0.0017, 0.0, (0.011 + 0.038) / 2),
    (0.0017, size[1] / 2, (0.038 - 0.011) / 2),
    friction=TRAY_FRICTION,
  )
  _box(
    body,
    f"{color}_bin_back",
    ((back - size[0] / 2) / 2, 0.0, size[2] / 2),
    ((back + size[0] / 2) / 2, size[1] / 2, size[2] / 2),
    friction=TRAY_FRICTION,
  )
  for side in ("left", "right"):
    body.add_geom(
      name=f"{color}_bin_wall_{side}",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=f"akrobin_wall_{side}",
      group=3,
      friction=TRAY_FRICTION,
    )


def _tray(root: mujoco.MjsBody) -> None:
  """The tray: a flat floor and, on each side, a flat top ``TRAY_LIP`` above the
  desk and a box whose upper face slopes from the top's inner edge into the floor.
  Both run the side's full length, so the corners overlap into a mitered basin."""
  tray = root.add_body(name="tray", pos=[*measures.TRAY_CENTER, desk.TOP])
  gray = [0.24, 0.24, 0.25, 1.0]
  half = np.array(measures.TRAY_SIZE) / 2
  rim = measures.TRAY_RIM
  lip = measures.TRAY_LIP
  floor = measures.TRAY_FLOOR_THICKNESS
  angle = measures.TRAY_WALL_ANGLE
  run = (lip - floor) / np.tan(angle)
  top = rim - run  # the flat top's width
  _box(
    tray,
    "tray_floor",
    (0, 0, floor - HALF_THICK),
    (half[0], half[1], HALF_THICK),
    rgba=gray,
    group=DRAWN,
    friction=TRAY_FRICTION,
  )
  up = np.array([0.0, 0.0, 1.0])
  for side, axis, sign in (
    ("near", 0, -1),
    ("far", 0, 1),
    ("left", 1, 1),
    ("right", 1, -1),
  ):
    out = np.zeros(3)
    out[axis] = sign  # the side's outward normal
    along = np.cross(up, out)
    length = half[1 - axis]
    edge = half[axis]
    center = out * (edge - top / 2) + up * (lip - HALF_THICK)
    size = np.abs(out) * top / 2 + np.abs(along) * length + up * HALF_THICK
    _box(tray, f"tray_top_{side}", center, size, rgba=gray, group=DRAWN)
    # The slope runs HALF_THICK past the floor so the two meet without a seam.
    down = -out * np.cos(angle) - up * np.sin(angle)
    normal = up * np.cos(angle) - out * np.sin(angle)
    start = out * (edge - top) + up * lip
    span = run / np.cos(angle) + HALF_THICK
    center = start + down * span / 2 - normal * HALF_THICK
    frame = np.stack([down, np.cross(normal, down), normal], axis=1)
    quat = rotation.from_matrix(frame)
    _box(
      tray,
      f"tray_slope_{side}",
      center,
      (span / 2, length, HALF_THICK),
      rgba=gray,
      group=DRAWN,
      quat=list(quat),
      friction=TRAY_FRICTION,
    )


def fixtures() -> mujoco.MjSpec:
  """The desk, the tray and the two AkroBins."""
  spec = mujoco.MjSpec()
  root = spec.worldbody.add_body(name="fixtures")
  desk.stand(root)
  _tray(root)
  _akrobin_meshes(spec)
  for color in measures.COLORS:
    body = root.add_body(
      name=f"{color}_bin",
      pos=[*measures.bin_middle(color), desk.TOP],
      quat=list(measures.BIN_TURN),
    )
    _akrobin(body, color)
  return spec


def brick(
  spec: mujoco.MjSpec, name: str, color: str, at: tuple[float, float, float]
) -> None:
  """Add one brick on a free joint, flat with its long side along x, its origin
  and mass at the middle of its collision box, studs included. ``at`` is where
  it stands in the reference pose, where the solver sizes its contact buffers."""
  length, width, height = measures.BRICK
  rise = 0.00185
  body_height = height - rise
  stud = 0.0049 / 2
  pitch = 0.008
  wall = 0.0012
  ceiling = 0.001
  sides = 12
  bottom = -height / 2.0
  top = bottom + body_height
  inside = bottom + body_height - ceiling
  vertices: list[tuple[float, float, float]] = []
  faces: list[tuple[int, int, int]] = []

  def points(*corners: tuple[float, float, float]) -> list[int]:
    vertices.extend(corners)
    return list(range(len(vertices) - len(corners), len(vertices)))

  def rectangle(z: float, inset: float) -> list[int]:
    x = length / 2.0 - inset
    y = width / 2.0 - inset
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

  outer_low = rectangle(bottom, 0.0)
  outer_high = rectangle(top, 0.0)
  inner_low = rectangle(bottom, wall)
  inner_high = rectangle(inside, wall)
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
    rgba=[*BRICK_RGB[color], 1.0],
    contype=0,
    conaffinity=0,
    density=0,
    group=DRAWN,
  )
  body.add_geom(
    name=f"{name}_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(length / 2.0, width / 2.0, height / 2.0),
    mass=measures.BRICK_MASS,
    group=3,
    friction=list(BRICK_FRICTION),
  )


def spec() -> mujoco.MjSpec:
  """The fixtures and every brick of ``BRICK_COLORS``, parked in its own bin."""
  room = fixtures()
  colors = measures.BRICK_COLORS
  for index, color in enumerate(colors):
    at = measures.parked(color, colors[:index].count(color))
    brick(room, f"brick_{index}", color, at)
  return room
