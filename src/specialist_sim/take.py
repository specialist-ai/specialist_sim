"""A solved take: each hand's images, pose and finger opening, to drive the env's
hands."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import av
import numpy as np
from specialist_dataset import schema
from specialist_dataset.read import Recording, read_anchors, read_lens

from specialist_sim import rotation
from specialist_sim.arrays import Float64Array, UInt8Array
from specialist_sim.camera.lens import Lens, Mei
from specialist_sim.env import Clock

# Which of a camera's two lenses a take was shot from.
SIDE: schema.LensName = "rear"


@dataclass(frozen=True)
class Track:
  """One hand's observations, and the state the policy is given beside them."""

  slot: str
  """The camera slot the state layer names for this hand."""
  tag_from_lens: Float64Array
  """The bundle-solved lens seat, a pose ``(7,)`` in the hand's tag frame."""
  times: Float64Array
  """When the layer reports a state."""
  poses: Float64Array
  """Each state's pose ``(m, 7)`` in the solve's world."""
  opening: Float64Array
  """The coupled finger angle at each of those times."""
  image_times: Float64Array
  """Each image's time in its own video, on a nominal-rate lattice; the two
  cameras' lattices sit a fixed offset apart for the session."""
  images: list[bytes]
  """One encoded image per image time."""


class Take:
  """One recorded session, one frame kept per control step of ``clock``."""

  def __init__(
    self, dataset: Path, session: str, clock: Clock, layer: str | None = None
  ):
    estimate = schema.layer_rrd(dataset, schema.ESTIMATE, session)
    self._base = schema.layer_rrd(dataset, schema.BASE, session)
    self._actions = clock.n_action_steps
    anchors = read_anchors(estimate).poses
    world_from_room = _world_from_room(anchors[min(anchors)], anchors[max(anchors)])
    self.room_from_world = rotation.inverse(world_from_room)
    observed = schema.layer_rrd(dataset, layer or _sole_layer(dataset), session)
    with Recording(observed) as seen, Recording(estimate) as solve:
      tracks = [_track(seen, solve, hand) for hand in schema.HANDS]
    image_times = np.mean([track.image_times for track in tracks], axis=0)
    stride = round(clock.step_dt / float(np.mean(np.diff(image_times))))
    self.tracks = tuple(
      replace(
        track, image_times=track.image_times[::stride], images=track.images[::stride]
      )
      for track in tracks
    )
    # A frame's nominal time, between the cameras that exposed it.
    self.image_times = image_times[::stride]
    self.period = float(np.mean(np.diff(self.image_times)))

  def lens(self, track: Track, width: int) -> Lens:
    """The lens that recorded a track: how it projects is what the base layer
    stores, where it sits is what the bundle solved."""
    recorded = read_lens(self._base, track.slot, SIDE, width)
    return Lens(
      params=Mei(**asdict(recorded.params)).scaled(1.0 / width),
      pos=tuple(track.tag_from_lens[:3]),
      quat=tuple(track.tag_from_lens[3:]),
    )

  def observed(self, index: int, frame: int) -> UInt8Array:
    """What hand ``index``'s camera saw at a frame."""
    context = av.CodecContext.create("mjpeg", "r")
    packet = av.Packet(self.tracks[index].images[frame])
    (decoded,) = [*context.decode(packet), *context.decode(None)]
    return np.asarray(decoded.to_ndarray(format="rgb24"), np.uint8)

  def action_chunk(self, frame: int) -> Float64Array:
    """The actions that land every gripper where the layer says it was at
    ``frame``, ``(n_action_steps, hands, 8)``, evenly over the frame before it."""
    ahead = (np.arange(1, self._actions + 1) / self._actions - 1.0) * self.period
    return np.stack(
      [
        states(track, self.room_from_world, track.image_times[frame] + ahead)
        for track in self.tracks
      ],
      axis=1,
    )

  def state(self, frame: int) -> Float64Array:
    """Where the layer says every gripper was at ``frame``, ``(hands, 8)``."""
    return np.stack(
      [
        states(track, self.room_from_world, track.image_times[frame : frame + 1])[0]
        for track in self.tracks
      ]
    )


def states(
  track: Track, room_from_world: Float64Array, when: Float64Array
) -> Float64Array:
  """One gripper's room-frame states at the times ``when``, ``(m, 8)``, read
  between the two samples each falls between."""
  upper = np.clip(np.searchsorted(track.times, when), 1, len(track.times) - 1)
  lower = upper - 1
  span = track.times[upper] - track.times[lower]
  blend = np.clip((when - track.times[lower]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
  out = np.empty((len(when), 8))
  out[:, 7] = np.interp(when, track.times, track.opening)
  for row, (low, high, fraction) in enumerate(zip(lower, upper, blend, strict=True)):
    world = rotation.interpolate(track.poses[low], track.poses[high], fraction)
    out[row, :7] = rotation.compose(room_from_world, world)
  return out


def _posed(translation: Float64Array, xyzw: Float64Array) -> Float64Array:
  """Stored translations ``(..., 3)`` and ``xyzw`` quaternions ``(..., 4)`` as
  poses ``(..., 7)``, their quaternions unit ``wxyz``."""
  wxyz = xyzw[..., [3, 0, 1, 2]]
  return np.concatenate(
    [translation, wxyz / np.linalg.norm(wxyz, axis=-1, keepdims=True)], axis=-1
  )


def _tag_seat(solve: Recording, slot: str) -> Float64Array:
  body_from_tag, body_from_lens = (
    _posed(
      np.asarray(row[schema.POSE_TRANSLATION], np.float64).reshape(3),
      np.asarray(row[schema.POSE_QUATERNION], np.float64).reshape(4),
    )
    for row in (
      solve.row(schema.base_tag(slot)),
      solve.row(schema.lens_seat(slot, SIDE)),
    )
  )
  return rotation.compose(rotation.inverse(body_from_tag), body_from_lens)


def _track(seen: Recording, solve: Recording, hand: str) -> Track:
  times, translations = seen.values(schema.state_pose(hand), schema.POSE_TRANSLATION, 3)
  stored = seen.values(schema.state_pose(hand), schema.POSE_QUATERNION, 4)[1]
  table = seen.series(f"/{schema.state_image(hand)}")
  column = schema.selector(schema.state_image(hand), "EncodedImage:blob")
  rowwise = table.column(column).combine_chunks().drop_null().to_pylist()
  slot = str(np.asarray(seen.row(schema.state_hand(hand))["slot"]).reshape(-1)[0])
  return Track(
    slot=slot,
    tag_from_lens=_tag_seat(solve, slot),
    times=times,
    poses=_posed(translations, stored),
    opening=seen.values(schema.state_opening(hand), schema.SCALARS, 1)[1][:, 0],
    image_times=seen.seconds(table),
    images=[bytes(blob) for (blob,) in rowwise],
  )


def _sole_layer(dataset: Path) -> str:
  """The one state configuration a dataset holds."""
  (found,) = sorted(p.name for p in dataset.glob(f"{schema.STATE}_*") if p.is_dir())
  return found


def _world_from_room(left: Float64Array, right: Float64Array) -> Float64Array:
  """Where the room sits in the take's world, a pose ``(7,)``, from the two
  anchors' ``(4, 4)`` poses. A solve is gravity-aligned: the anchors' midpoint
  is the origin, and the level line from the right anchor to the left is +y."""
  across = left[:3, 3] - right[:3, 3]
  across[2] = 0.0
  across = across / np.linalg.norm(across)
  up = np.array([0.0, 0.0, 1.0])
  return np.concatenate(
    [
      (left[:3, 3] + right[:3, 3]) / 2.0,
      rotation.from_matrix(np.stack([np.cross(across, up), across, up], axis=1)),
    ]
  )
