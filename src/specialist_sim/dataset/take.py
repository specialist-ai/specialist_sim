"""A solved take, read as the state a policy is trained on: each hand's
images, pose, twist and finger opening, in the solve's world.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from specialist_corpus import schema
from specialist_corpus.poses import quats_to_matrices
from specialist_corpus.read import Recording, read_anchors, read_lens

from specialist_sim import rotation
from specialist_sim.action import CHUNK_SIZE
from specialist_sim.arrays import Float64Array, UInt8Array
from specialist_sim.camera import Camera, Mei
from specialist_sim.dataset.layer import SIDE, decoded

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
  poses: Float64Array
  """Each state's ``(4, 4)`` pose in the solve's world."""
  opening: Float64Array
  """The coupled finger angle at each of those times."""
  ticks: Float64Array
  """Each frame's time in its own video, a lattice at the camera's nominal
  rate; the two cameras' lattices sit a fixed offset apart for the session."""
  frames: list[bytes]
  """One encoded image per tick."""


class Take:
  """One recorded session. ``control_hz`` keeps one frame per control step;
  ``None`` keeps every frame the layer holds."""

  def __init__(
    self,
    dataset: Path,
    session: str,
    layer: str | None = None,
    control_hz: float | None = None,
  ):
    estimate = schema.layer_rrd(dataset, schema.ESTIMATE, session)
    self.base = schema.layer_rrd(dataset, schema.BASE, session)
    anchors = read_anchors(estimate).poses
    self.room_from_world = _inverse(
      _world_from_room(anchors[min(anchors)], anchors[max(anchors)])
    )
    observed = schema.layer_rrd(dataset, layer or _sole_layer(dataset), session)
    with Recording(observed) as seen:
      slots = {hand: _slot(seen, hand) for hand in schema.HANDS}
      with Recording(estimate) as solve:
        seats = {hand: _tag_seat(solve, slot) for hand, slot in slots.items()}
      self.tracks = tuple(
        _track(seen, hand, slots[hand], seats[hand]) for hand in schema.HANDS
      )
    if control_hz is not None:
      self.tracks = _thinned(self.tracks, control_hz)
    self.times = np.mean([track.ticks for track in self.tracks], axis=0)
    """A tick's nominal time, between the cameras that exposed it."""
    self.period = float(np.mean(np.diff(self.times)))

  def camera(self, track: Track, width: int) -> Camera:
    """The camera that recorded a track: how the lens projects is what the
    base layer stores, where it sits is what the bundle solved."""
    lens = read_lens(self.base, track.slot, SIDE, width)
    return Camera(
      mei=Mei(**asdict(lens.params)),
      width=width,
      pos=tuple(track.tag_from_lens[:3, 3]),
      quat=tuple(rotation.from_matrix(track.tag_from_lens[:3, :3])),
    )

  def observed(self, hand: int, tick: int) -> UInt8Array:
    """What one camera saw at a tick, as the policy is trained to see it."""
    return decoded(self.tracks[hand].frames[tick])

  def action_chunk(self, tick: int, n_action_steps: int) -> Float64Array:
    """The action chunk that lands every gripper where the layer says it was at
    ``tick``, ``(CHUNK_SIZE, hands, 8)``, the first ``n_action_steps`` of them
    leading up to it."""
    ahead = (np.arange(1, CHUNK_SIZE + 1) / n_action_steps - 1.0) * self.period
    return np.stack(
      [self.poses(track, track.ticks[tick] + ahead) for track in self.tracks], axis=1
    )

  def pose(self, tick: int) -> Float64Array:
    """Where the layer says every gripper was at ``tick``, ``(hands, 8)``."""
    return np.stack(
      [self.poses(track, track.ticks[tick : tick + 1])[0] for track in self.tracks]
    )

  def poses(self, track: Track, when: Float64Array) -> Float64Array:
    """One gripper's room-frame states at the named times, ``(n, 8)``, read
    between the two samples each falls between."""
    upper = np.clip(np.searchsorted(track.times, when), 1, len(track.times) - 1)
    lower = upper - 1
    span = track.times[upper] - track.times[lower]
    blend = np.clip(
      (when - track.times[lower]) / np.where(span > 0, span, 1.0), 0.0, 1.0
    )
    out = np.zeros((len(when), 8))
    for row in range(len(when)):
      low, high, fraction = lower[row], upper[row], blend[row]
      position = (1.0 - fraction) * track.poses[low, :3, 3] + (
        fraction * track.poses[high, :3, 3]
      )
      start = rotation.from_matrix(track.poses[low, :3, :3])
      end = rotation.from_matrix(track.poses[high, :3, :3])
      spin = rotation.log(rotation.multiply(rotation.conjugate(start), end))
      between = rotation.multiply(start, rotation.exp(fraction * spin))
      world = np.eye(4)
      world[:3, :3] = rotation.to_matrix(between)
      world[:3, 3] = position
      gripper = self.room_from_world @ world
      out[row, :3] = gripper[:3, 3]
      out[row, 3:7] = rotation.from_matrix(gripper[:3, :3])
      out[row, 7] = float(np.interp(when[row], track.times, track.opening))
    return out


def _slot(seen: Recording, hand: str) -> str:
  """Which camera rode one hand."""
  return str(np.asarray(seen.row(schema.state_hand(hand))["slot"]).reshape(-1)[0])


def _pose(row: dict[str, object]) -> Float64Array:
  """One entity's stored transform."""
  out = np.eye(4)
  out[:3, :3] = quats_to_matrices(
    np.asarray(row[schema.POSE_QUATERNION]).reshape(1, 4)
  )[0]
  out[:3, 3] = np.asarray(row[schema.POSE_TRANSLATION]).reshape(3)
  return out


def _tag_seat(solve: Recording, slot: str) -> Float64Array:
  """Where the bundle solved the rear lens sits, in the tracked frame."""
  body_from_tag = _pose(solve.row(schema.base_tag(slot)))
  body_from_lens = _pose(solve.row(schema.lens_seat(slot, SIDE)))
  return _inverse(body_from_tag) @ body_from_lens


def _track(seen: Recording, hand: str, slot: str, tag_from_lens: Float64Array) -> Track:
  times, translations = seen.values(schema.state_pose(hand), schema.POSE_TRANSLATION, 3)
  stored = seen.values(schema.state_pose(hand), schema.POSE_QUATERNION, 4)[1]
  poses = np.tile(np.eye(4), (len(times), 1, 1))
  poses[:, :3, :3] = quats_to_matrices(stored)
  poses[:, :3, 3] = translations
  table = seen.series(f"/{schema.state_image(hand)}")
  column = schema.selector(schema.state_image(hand), _BLOB)
  rowwise = table.column(column).combine_chunks().drop_null().to_pylist()
  return Track(
    hand=hand,
    slot=slot,
    tag_from_lens=tag_from_lens,
    times=times,
    poses=poses,
    opening=seen.values(schema.state_opening(hand), schema.SCALARS, 1)[1][:, 0],
    ticks=seen.seconds(table),
    frames=[bytes(blob) for (blob,) in rowwise],
  )


def _thinned(tracks: tuple[Track, ...], control_hz: float) -> tuple[Track, ...]:
  """The tracks' frames kept at the rate a control step runs at."""
  shared = np.mean([track.ticks for track in tracks], axis=0)
  stride = round(1.0 / (control_hz * float(np.mean(np.diff(shared)))))
  return tuple(
    replace(track, ticks=track.ticks[::stride], frames=track.frames[::stride])
    for track in tracks
  )


def _sole_layer(dataset: Path) -> str:
  """The one state configuration a dataset holds."""
  (found,) = sorted(p.name for p in dataset.glob(f"{schema.STATE}_*") if p.is_dir())
  return found


def _inverse(pose: Float64Array) -> Float64Array:
  out = np.eye(4)
  out[:3, :3] = pose[:3, :3].T
  out[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
  return out


def _world_from_room(left: Float64Array, right: Float64Array) -> Float64Array:
  """Where the room sits in the take's world. A solve is gravity-aligned: the
  anchors' midpoint fixes the origin, and the line between them fixes across."""
  across = left[:3, 3] - right[:3, 3]
  across[2] = 0.0
  across = across / np.linalg.norm(across)
  up = np.array([0.0, 0.0, 1.0])
  out = np.eye(4)
  out[:3, :3] = np.stack([np.cross(across, up), across, up], axis=1)
  out[:3, 3] = (left[:3, 3] + right[:3, 3]) / 2.0
  return out
