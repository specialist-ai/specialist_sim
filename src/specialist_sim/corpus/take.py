"""A solved take, read as the state a policy is trained on.

The state layer holds, for each hand, the images the policy sees and the
gripper's own pose, twist and finger opening. Those poses are in the solve's
world, so the anchors the solve held fixed say where the room is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import av
import mujoco
import numpy as np
from specialist_corpus import schema
from specialist_corpus.poses import quats_to_matrices
from specialist_corpus.read import Recording, read_anchors, read_lens

from specialist_sim import pose as se3
from specialist_sim.action import slerp
from specialist_sim.arrays import Float64Array, Int64Array, UInt8Array
from specialist_sim.calibration import CameraCfg
from specialist_sim.mei import MeiParams
from specialist_sim.spaces import HORIZON, PER_HAND

SIDE: schema.Side = "rear"
"""The lens the pipeline reads: the rear camera on each gripper."""

__all__ = ["Take", "Track"]

_BLOB = "EncodedImage:blob"


@dataclass(frozen=True)
class Track:
  """One hand's observations, and the state the policy is given beside them."""

  hand: str
  slot: str
  """The camera that rode this hand."""
  tag_from_lens: Float64Array
  """Where the bundle solved the lens sits, in the frame the hand is tracked by."""
  times: Float64Array
  """When the layer reports a state, which is finer than it reports a frame."""
  translations: Float64Array
  quaternions: Float64Array
  """Each pose's rotation, in the order the tracker's slerp reads."""
  speeds: Float64Array
  """The gripper frame's world velocity at each of those times."""
  opening: tuple[Float64Array, Float64Array]
  """The coupled finger angle over time."""
  ticks: Float64Array
  """When this camera exposed each image. The hands share the step lattice,
  not the instant: their shutters sit a fixed offset apart."""
  frames: list[bytes]
  """One encoded image per tick."""


class Take:
  """One recorded session, on the clock its observations were written to."""

  def __init__(
    self,
    dataset: Path,
    session: str,
    layer: str | None = None,
    action_hz: float = 50.0,
  ):
    self.action_hz = action_hz
    """How often the trajectory this take hands back names a pose. It has to be
    the rate of the environment the take drives, which is ``PickPlaceCfg``'s."""
    estimate = schema.layer_rrd(dataset, schema.ESTIMATE, session)
    self.base = schema.layer_rrd(dataset, schema.BASE, session)
    anchors = read_anchors(estimate).poses
    if len(anchors) < 2:
      raise ValueError(f"{session} solved {len(anchors)} anchors, and a room needs two")
    self.room_from_world = se3.inverse(
      _world_from_room(anchors[min(anchors)], anchors[max(anchors)])
    )
    observed = schema.layer_rrd(dataset, layer or _sole_layer(dataset), session)
    with Recording(observed) as seen:
      self.steps = _steps(seen)
      slots = {hand: _slot(seen, hand) for hand in schema.HANDS}
      with Recording(estimate) as solve:
        seats = {hand: _tag_seat(solve, slot) for hand, slot in slots.items()}
      self.tracks = tuple(
        _track(seen, hand, slots[hand], seats[hand]) for hand in schema.HANDS
      )
    self.times = np.mean([track.ticks for track in self.tracks], axis=0)
    """A tick's nominal time, between the cameras that exposed it."""
    self.start = float(self.times[0])
    self.stop = float(self.times[-1])
    self.period = float(np.mean(np.diff(self.times)))

  def camera(self, track: Track, width: int) -> CameraCfg:
    """The camera that recorded a track, as the environment configures one.

    How the lens projects is what the base layer stores for it; where it sits
    on the gripper is what the bundle solved.
    """
    lens = read_lens(self.base, track.slot, SIDE, width)
    return CameraCfg(
      params=MeiParams(**asdict(lens.params)),
      width=width,
      pos=tuple(track.tag_from_lens[:3, 3]),
      quat=tuple(se3.wxyz(track.tag_from_lens[:3, :3])),
    )

  def observed(self, hand: int, tick: int) -> UInt8Array:
    """What one camera saw at a tick, as the policy is trained to see it."""
    return _decoded(self.tracks[hand].frames[tick])

  def tick_at(self, when: float) -> int:
    """The tick whose frame sits nearest a time on the session clock."""
    return int(np.argmin(np.abs(self.times - when)))

  def trajectory(self, tick: int) -> Float64Array:
    """What lands every gripper where the layer says it was at ``tick``.

    A control step tracks a trajectory forward by one tick, so it starts a tick
    back and the last pose the step tracks falls on the tick itself.
    """
    ahead = np.arange(1, HORIZON + 1) / self.action_hz - self.period
    out = np.zeros((HORIZON, PER_HAND * len(self.tracks)))
    for hand, track in enumerate(self.tracks):
      at = hand * PER_HAND
      out[:, at : at + PER_HAND] = self._poses(track, track.ticks[tick] + ahead)
    return out

  def pose(self, tick: int) -> Float64Array:
    """Where the layer says every gripper was at ``tick``."""
    return np.concatenate(
      [self._poses(track, track.ticks[tick : tick + 1])[0] for track in self.tracks]
    )

  def speed(self, tick: int) -> Float64Array:
    """How fast each gripper was moving at ``tick``."""
    when = self.times[tick]
    return np.array(
      [
        float(
          np.linalg.norm(
            [np.interp(when, track.times, track.speeds[:, axis]) for axis in range(3)]
          )
        )
        for track in self.tracks
      ]
    )

  def _poses(self, track: Track, when: Float64Array) -> Float64Array:
    """One gripper's room-frame poses at the named times.

    The layer reports a state every ``state_period``, which is coarser than a
    trajectory names one, so a pose is read between the two samples it falls
    between rather than off the nearer of them.
    """
    upper = np.clip(np.searchsorted(track.times, when), 1, len(track.times) - 1)
    lower = upper - 1
    span = track.times[upper] - track.times[lower]
    blend = np.clip(
      (when - track.times[lower]) / np.where(span > 0, span, 1.0), 0.0, 1.0
    )
    times, opening = track.opening
    out = np.zeros((len(when), PER_HAND))
    spin = np.empty(3)
    for row in range(len(when)):
      low, high, fraction = lower[row], upper[row], blend[row]
      position = (1.0 - fraction) * track.translations[low] + (
        fraction * track.translations[high]
      )
      mujoco.mju_subQuat(spin, track.quaternions[high], track.quaternions[low])
      between = slerp(track.quaternions[low], spin, np.array([fraction]))[0]
      gripper = self.room_from_world @ se3.of(se3.rotation(between), position)
      out[row, :3] = gripper[:3, 3]
      out[row, 3:7] = se3.wxyz(gripper[:3, :3])
      out[row, 7] = float(np.interp(when[row], times, opening))
    return out


def _steps(seen: Recording) -> Int64Array:
  """The lattice every hand's images share, which their instants do not."""
  lattices = [_lattice(seen, schema.state_image(hand)) for hand in schema.HANDS]
  if any(not np.array_equal(lattices[0], other) for other in lattices[1:]):
    raise ValueError(f"{seen.path}: the hands' images sit on different lattices")
  if not len(lattices[0]):
    raise ValueError(f"{seen.path} holds no images")
  return lattices[0]


def _lattice(seen: Recording, entity: str) -> Int64Array:
  """One entity's step index, the integer row of the state stream."""
  column = seen.series(f"/{entity}").column(schema.ROW_TIMELINE)
  out: Int64Array = np.asarray(column.combine_chunks().to_pylist(), np.int64)
  return out


def _slot(seen: Recording, hand: str) -> str:
  """Which camera rode one hand."""
  return str(np.asarray(seen.row(schema.state_hand(hand))["slot"]).reshape(-1)[0])


def _pose(row: dict[str, object]) -> Float64Array:
  """One entity's stored transform."""
  return se3.of(
    quats_to_matrices(np.asarray(row[schema.POSE_QUATERNION]).reshape(1, 4))[0],
    np.asarray(row[schema.POSE_TRANSLATION]).reshape(3),
  )


def _tag_seat(solve: Recording, slot: str) -> Float64Array:
  """Where the bundle solved the rear lens sits, in the tracked frame."""
  body_from_tag = _pose(solve.row(schema.base_tag(slot)))
  body_from_lens = _pose(solve.row(schema.lens_seat(slot, SIDE)))
  return se3.inverse(body_from_tag) @ body_from_lens


def _track(seen: Recording, hand: str, slot: str, tag_from_lens: Float64Array) -> Track:
  """Everything one hand's entities hold."""
  times, translations = seen.values(schema.state_pose(hand), schema.POSE_TRANSLATION, 3)
  stored = seen.values(schema.state_pose(hand), schema.POSE_QUATERNION, 4)[1]
  quaternions = np.stack([se3.wxyz(m) for m in quats_to_matrices(stored)])
  speeds = seen.values(schema.state_lin_vel(hand), schema.SCALARS, 3)[1]
  times, opening = seen.values(schema.state_opening(hand), schema.SCALARS, 1)
  table = seen.series(f"/{schema.state_image(hand)}")
  column = schema.selector(schema.state_image(hand), _BLOB)
  rowwise = table.column(column).combine_chunks().drop_null().to_pylist()
  ticks = seen.seconds(table)
  return Track(
    hand=hand,
    slot=slot,
    tag_from_lens=tag_from_lens,
    times=times,
    translations=translations,
    quaternions=quaternions,
    speeds=speeds,
    opening=(times, opening[:, 0]),
    ticks=ticks,
    frames=[bytes(blob) for (blob,) in rowwise],
  )


def _sole_layer(dataset: Path) -> str:
  """The one state configuration a dataset holds."""
  found = sorted(p.name for p in dataset.glob(f"{schema.STATE}_*") if p.is_dir())
  if len(found) != 1:
    raise ValueError(f"{dataset} holds {found}, so name the one to read")
  return found[0]


def _decoded(blob: bytes) -> UInt8Array:
  """One encoded image as pixels."""
  context = av.CodecContext.create("png", "r")
  decoded = [*context.decode(av.Packet(blob)), *context.decode(None)]
  if len(decoded) != 1:
    raise ValueError(f"an image holds {len(decoded)} frames, not one")
  out: UInt8Array = np.asarray(decoded[0].to_ndarray(format="rgb24"), np.uint8)
  return out


def _world_from_room(left: Float64Array, right: Float64Array) -> Float64Array:
  """Where the room sits in the take's world.

  A solve is gravity-aligned: the anchors' midpoint fixes the origin, and the
  line between them fixes across.
  """
  across = left[:3, 3] - right[:3, 3]
  across[2] = 0.0
  reach = float(np.linalg.norm(across))
  if reach < 1e-6:
    raise ValueError("the anchors stand on top of each other, leaving no yaw")
  across = across / reach
  up = np.array([0.0, 0.0, 1.0])
  rotation = np.stack([np.cross(across, up), across, up], axis=1)
  return se3.of(rotation, (left[:3, 3] + right[:3, 3]) / 2.0)
