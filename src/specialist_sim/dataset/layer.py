"""An episode written as a state layer, in the shape a recorded one has:
every frame the lenses exposed, and each hand's pose, twist and opening at
every row, with the cube's. Everything is in the room frame.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import rerun as rr
from rerun.experimental import Chunk, LazyChunkStream, OptimizationProfile, send_chunks
from specialist_corpus import schema

from specialist_sim.arrays import Float64Array, Int64Array, UInt8Array
from specialist_sim.camera import NOMINAL
from specialist_sim.expert import Expert
from specialist_sim.pickplace import PickPlace, Report
from specialist_sim.pickplace.room import ASSETS

__all__ = [
  "CUBE_POSE",
  "SIDE",
  "Episode",
  "decoded",
  "encode",
  "record",
  "shutters",
  "revision",
  "scale",
  "write",
]

SIDE: schema.Side = "rear"
"""The lens a layer's images come from: the rear camera on each gripper."""

SLOTS = tuple(f"cam{index:02d}" for index in range(len(schema.HANDS)))
"""The handle a capture gives each camera, which a layer names its hands by."""

CUBE_POSE = f"{schema.STATE}/cube/pose"
"""Where the cube's pose is written; only a simulated scene knows what the
hands are acting on."""


@dataclass
class Episode:
  """One rollout, sampled as a capture samples it."""

  row_hz: float
  row_times: list[float] = field(default_factory=list)
  poses: dict[str, list[Float64Array]] = field(default_factory=dict)
  twists: dict[str, list[Float64Array]] = field(default_factory=dict)
  openings: dict[str, list[float]] = field(default_factory=dict)
  cube: list[Float64Array] = field(default_factory=list)
  frame_times: dict[str, list[float]] = field(default_factory=dict)
  frames: dict[str, list[bytes]] = field(default_factory=dict)

  def __post_init__(self) -> None:
    for hand in schema.HANDS:
      self.frame_times.setdefault(hand, [])
      self.poses.setdefault(hand, [])
      self.twists.setdefault(hand, [])
      self.openings.setdefault(hand, [])
      self.frames.setdefault(hand, [])


def record(
  env: PickPlace, expert: Expert, seed: int, max_steps: int
) -> tuple[Episode, Report]:
  """Drive one episode until the demonstration is finished, keeping what a
  state layer holds. Every hand carries a lens, as a capture's does."""
  clock, rig, room = env.cfg.clock, env.rig, env.room
  episode = Episode(1.0 / clock.row_dt)
  observation, report = env.reset([seed])
  for _ in range(max_steps):
    action = expert.act(observation.state[0, -1], report.cube[0])
    observation, report = env.step(action[None])
    states = rig.state_rows[0]
    twists = rig.twists(rig.qpos_rows[0], rig.qvel_rows[0])
    for index, row in enumerate(observation.row[0]):
      episode.row_times.append(float(row * clock.row_dt))
      for hand, name in enumerate(schema.HANDS):
        episode.poses[name].append(states[index, hand, :7])
        episode.twists[name].append(twists[index, hand])
        episode.openings[name].append(float(states[index, hand, 7]))
      at = room.cube.qpos
      episode.cube.append(rig.qpos_rows[0, index, at : at + 7].copy())
    for name, slot in zip(schema.HANDS, rig.slots, strict=True):
      for frame in rig.exposed[0][slot]:
        episode.frame_times[name].append(frame.at * clock.row_dt)
        episode.frames[name].append(encode(frame.image))
    if expert.done or report.dropped[0]:
      break
  paired = min(len(episode.frames[name]) for name in schema.HANDS)
  for name in schema.HANDS:
    del episode.frames[name][paired:]
    del episode.frame_times[name][paired:]
  return episode, report


def encode(image: UInt8Array) -> bytes:
  """One frame as the PNG a layer stores."""
  context = av.CodecContext.create("png", "w")
  context.width, context.height, context.pix_fmt = (
    image.shape[1],
    image.shape[0],
    "rgb24",
  )
  frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
  (packet,) = [*context.encode(frame), *context.encode(None)]
  return bytes(packet)


def decoded(blob: bytes) -> UInt8Array:
  """One encoded image as pixels."""
  context = av.CodecContext.create("png", "r")
  (frame,) = [*context.decode(av.Packet(blob)), *context.decode(None)]
  out: UInt8Array = np.asarray(frame.to_ndarray(format="rgb24"), np.uint8)
  return out


def scale(size: int) -> int:
  """How many times smaller than the nominal lens a layer's images are."""
  return NOMINAL.width // size


def revision() -> str:
  """The commit the demonstration was written by, read from the history the
  package sits in."""
  root = Path(__file__).resolve().parents[3]
  found = subprocess.run(
    ["git", "-C", str(root), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=True,
  )
  return found.stdout.strip()


def _model_sha() -> str:
  """What the room the demonstration ran in is, as a capture names its URDF."""
  digest = hashlib.sha1()
  for asset in sorted(path for path in ASSETS.rglob("*") if path.is_file()):
    digest.update(asset.relative_to(ASSETS).as_posix().encode())
    digest.update(asset.read_bytes())
  return digest.hexdigest()


def _rows_at(times: Float64Array, row_hz: float) -> Int64Array:
  out: Int64Array = np.rint(times * row_hz).astype(np.int64)
  return out


def _clock(times: Float64Array, rows: Int64Array) -> list[Any]:
  """The session clock at the instants a series was measured, and the row
  each is filed under."""
  return [
    rr.TimeColumn(
      schema.TIMELINE,
      duration=np.rint(times * schema.NS_PER_SECOND).astype("timedelta64[ns]"),
    ),
    rr.TimeColumn(schema.ROW_TIMELINE, sequence=rows),
  ]


def shutters(episode: Episode) -> Int64Array:
  """The row a pair of exposures is filed under: the first hand's."""
  return _rows_at(np.asarray(episode.frame_times[schema.HANDS[0]]), episode.row_hz)


def _hand_chunks(
  episode: Episode, hand: str, slot: str, size: int, taken: Int64Array
) -> list[Chunk]:
  rows = np.asarray(episode.row_times)
  lattice = _clock(rows, _rows_at(rows, episode.row_hz))
  poses = np.asarray(episode.poses[hand])
  twists = np.asarray(episode.twists[hand])
  frames = np.asarray(episode.frame_times[hand])
  own = _rows_at(frames, episode.row_hz)
  blobs = episode.frames[hand]
  # One image plane holds both hands, left then right.
  column = float(schema.HANDS.index(hand) * size)
  return [
    Chunk.from_columns(
      schema.state_hand(hand), indexes=[], columns=rr.AnyValues.columns(slot=[slot])
    ),
    Chunk.from_columns(
      schema.state_image(hand),
      indexes=[],
      columns=rr.Transform3D.columns(translation=[[column, 0.0, 0.0]]),
    ),
    Chunk.from_columns(
      schema.state_image(hand),
      indexes=_clock(frames, taken),
      columns=[
        *rr.EncodedImage.columns(blob=blobs, media_type=["image/png"] * len(blobs)),
        *rr.AnyValues.columns(capture_row=own),
      ],
    ),
    Chunk.from_columns(
      schema.state_pose(hand),
      indexes=lattice,
      columns=rr.Transform3D.columns(
        translation=poses[:, :3], quaternion=poses[:, [4, 5, 6, 3]]
      ),
    ),
    Chunk.from_columns(
      schema.state_lin_vel(hand),
      indexes=lattice,
      columns=rr.Scalars.columns(scalars=twists[:, :3]),
    ),
    Chunk.from_columns(
      schema.state_ang_vel(hand),
      indexes=lattice,
      columns=rr.Scalars.columns(scalars=twists[:, 3:]),
    ),
    Chunk.from_columns(
      schema.state_opening(hand),
      indexes=lattice,
      columns=rr.Scalars.columns(scalars=np.asarray(episode.openings[hand])[:, None]),
    ),
  ]


def cube_chunk(episode: Episode) -> Chunk:
  """Where the cube stood at every row, as a pose in the room frame, in the
  ``xyzw`` order the dataset reads."""
  poses = np.asarray(episode.cube)
  rows = np.asarray(episode.row_times)
  return Chunk.from_columns(
    CUBE_POSE,
    indexes=_clock(rows, _rows_at(rows, episode.row_hz)),
    columns=rr.Transform3D.columns(
      translation=poses[:, :3], quaternion=poses[:, [4, 5, 6, 3]]
    ),
  )


def write(
  path: Path,
  session: str,
  episode: Episode,
  size: int,
  seed: int,
  sha: str,
  camera_hz: float,
) -> None:
  """Write one episode under its session id, in the layer's own shape. Its
  account names the seed that reproduces it, where a capture's names a solve."""
  taken = shutters(episode)
  chunks = [
    chunk
    for hand, slot in zip(schema.HANDS, SLOTS, strict=True)
    for chunk in _hand_chunks(episode, hand, slot, size, taken)
  ]
  chunks.append(cube_chunk(episode))
  written = datetime.now(timezone.utc).isoformat(timespec="seconds")
  rows = episode.row_times
  account = {
    "config": path.parent.name,
    "f_row": episode.row_hz,
    "f_image": camera_hz,
    "scale": scale(size),
    "mask_radius": NOMINAL.mask,
    "side": SIDE,
    "duration": rows[-1] - rows[0] if rows else 0.0,
    "git_sha": sha,
    "estimate_git_sha": sha,
    "urdf_sha": _model_sha(),
    "rerun_version": rr.__version__,
    "converted_at": int(datetime.now(timezone.utc).timestamp() * 1e9),
    "estimated_at": written,
    "written_at": written,
    "seed": seed,
  }
  path.parent.mkdir(parents=True, exist_ok=True)
  partial = path.parent / (path.name + ".partial")
  store = LazyChunkStream.from_iter(chunks).collect(
    optimize=OptimizationProfile.OBJECT_STORE
  )
  try:
    with rr.RecordingStream(
      schema.APPLICATION_ID, recording_id=session, send_properties=False
    ) as stream:
      stream.set_sinks(rr.FileSink(partial))
      values = cast("Any", rr.AnyValues)
      stream.send_property(schema.STATE, values(**account))
      send_chunks(store, recording=stream)
    partial.replace(path)
  finally:
    partial.unlink(missing_ok=True)
