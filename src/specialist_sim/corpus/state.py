"""An episode written as a state layer, in the shape a recorded one has.

The plant is sampled on the same lattice a capture is: images at the rate the
policy sees them, and the pose, twist and opening of each hand at ``f_row``,
together with the pose of the cube they act on, so a reader can stand the whole
scene back up rather than run an episode again to approximate it.
Everything is already in the room frame, which is the frame a solve's world
becomes once its anchors are read, so nothing is composed on the way out.
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

from specialist_sim import PickPlaceEnv
from specialist_sim.arrays import Float64Array, UInt8Array
from specialist_sim.corpus.take import SIDE
from specialist_sim.observation import flat_row
from specialist_sim.spaces import GRIPPERS, MASK_RADIUS

__all__ = [
  "Episode",
  "IMAGE_STRIDE",
  "layer_scale",
  "revision",
  "write_layer",
]


CAPTURE_FPS = 30.0
"""How fast the capture a demonstration stands in for runs."""
IMAGE_STRIDE = 3
"""Capture frames per stored image. The environment renders the frames that are
kept and no others, and they land where the capture's keyframes do."""

SLOTS = tuple(f"cam{index:02d}" for index in range(len(GRIPPERS)))
"""The handle a capture gives each camera, which a layer names its hands by."""

CUBE_POSE = f"{schema.STATE}/cube/pose"
"""Where the cube's pose is written. A recording holds no such entity: only a
simulated scene knows what the hands are acting on."""


def layer_scale(width: int, size: int) -> int:
  """Lens pixels per stored pixel. The environment renders at the stored size,
  so this is how far the intrinsics were scaled down to get there."""
  scale = width / size
  if not float(scale).is_integer():
    raise ValueError(f"a lens {width} across does not divide into {size}-pixel images")
  return int(scale)


@dataclass
class Episode:
  """One rollout, sampled as a capture samples it."""

  f_row: float = 100.0
  """The lattice the episode was sampled on, which is the environment's own.
  A reader indexes the layer by it, so it travels with the episode rather than
  being a constant every writer has to agree on."""
  step_times: list[float] = field(default_factory=list)
  poses: dict[str, list[Float64Array]] = field(default_factory=dict)
  twists: dict[str, list[Float64Array]] = field(default_factory=dict)
  openings: dict[str, list[float]] = field(default_factory=dict)
  cube: list[Float64Array] = field(default_factory=list)
  frame_times: list[float] = field(default_factory=list)
  frames: dict[str, list[bytes]] = field(default_factory=dict)

  def __post_init__(self) -> None:
    for hand in schema.HANDS:
      self.poses.setdefault(hand, [])
      self.twists.setdefault(hand, [])
      self.openings.setdefault(hand, [])
      self.frames.setdefault(hand, [])


def rollout(env: PickPlaceEnv, policy: Any, seed: int, max_steps: int) -> Episode:
  """Drive one episode, keeping what a state layer holds.

  A demonstration runs to the end of what it set out to do, so the episode ends
  when the policy is finished rather than when the task first reports success.
  The plant is read off ``qpos`` and ``qvel``, which a physics step leaves
  current, so sampling costs no extra solve.
  """
  kept = CAPTURE_FPS / IMAGE_STRIDE
  if not np.isclose(env.cfg.control_hz, kept):
    raise ValueError(
      f"a capture at {CAPTURE_FPS} fps keeps an image every {IMAGE_STRIDE} frames, "
      f"which is {kept} Hz, and the environment renders at {env.cfg.control_hz}"
    )
  if env.worlds != 1:
    raise ValueError(
      f"a session records one world, and this environment has {env.worlds}"
    )
  observation, info = env.reset(seed=seed)
  record = _Recorder(env)
  for _ in range(max_steps):
    record.frame(observation)
    action = policy.act(flat_row(observation), info["cube_pos"][0])
    observation, _, _, _, info = env.step(action, watch=record.watch)
    record.tick()
    if policy.done or info["dropped"]:
      break
  record.frame(observation)
  return record.episode


class _Recorder:
  """One world's episode, sampled on the lattice a capture samples it on."""

  def __init__(self, env: PickPlaceEnv):
    f_row = env.cfg.f_row
    self.episode = Episode(f_row=f_row)
    self.scene = env.scene
    self.clock = 0.0
    self._physics_dt = env.cfg.scene.physics_dt
    self._control_dt = 1.0 / env.cfg.control_hz
    self._every = round(1.0 / (f_row * self._physics_dt))
    if not np.isclose(self._every * self._physics_dt * f_row, 1.0):
      raise ValueError(
        f"a physics step of {self._physics_dt} does not divide the "
        f"{f_row} Hz lattice, and a stamp is never bent to fit"
      )

  def tick(self) -> None:
    """A control step has run; the clock stands at its boundary."""
    self.clock += self._control_dt

  def frame(self, observation: dict[str, Any]) -> None:
    """Keep what the cameras delivered at the boundary the clock stands on."""
    _aligned(observation)
    episode = self.episode
    episode.frame_times.append(self.clock)
    for hand, slot in zip(schema.HANDS, GRIPPERS, strict=True):
      episode.frames[hand].append(encode(observation[slot]["image"][0]))

  def watch(self, substep: int, qpos: Float64Array, qvel: Float64Array) -> None:
    """Keep the plant on every lattice row a control step crosses."""
    if (substep + 1) % self._every:
      return
    episode, scene = self.episode, self.scene
    episode.step_times.append(self.clock + (substep + 1) * self._physics_dt)
    seen, moving = scene.observe(qpos[:1])[0], scene.twists(qpos[:1], qvel[:1])[0]
    for index, hand in enumerate(schema.HANDS):
      episode.poses[hand].append(seen[index, :7].copy())
      episode.twists[hand].append(moving[index].copy())
      episode.openings[hand].append(float(seen[index, 7]))
    cube = scene.cube.qpos
    episode.cube.append(np.array(qpos[0, cube : cube + 7], np.float64))


def _aligned(observation: dict[str, Any]) -> None:
  """A layer stamps one row on a state and on the image beside it, so the two
  have to have been taken at the same one."""
  row = int(observation["row"][0, -1])
  for slot in GRIPPERS:
    taken = int(observation[slot]["image_row"][0])
    if taken != row:
      raise ValueError(
        f"a layer holds one clock: the state sits on row {row} and {slot}'s "
        f"image was taken at {taken}"
      )


def encode(image: UInt8Array) -> bytes:
  """One frame as the PNG a layer stores."""
  context = av.CodecContext.create("png", "w")
  context.width, context.height, context.pix_fmt = (
    image.shape[1],
    image.shape[0],
    "rgb24",
  )
  frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
  packets = [*context.encode(frame), *context.encode(None)]
  if len(packets) != 1:
    raise ValueError(f"a frame encoded to {len(packets)} packets, not one")
  return bytes(packets[0])


def revision() -> str:
  """The commit the demonstration was written by.

  A tree copied to wherever the episodes are generated carries no history, so
  this raises rather than record an unknown provenance: the caller passes the
  revision it read where the history is.
  """
  root = Path(__file__).resolve().parent.parent.parent.parent
  found = subprocess.run(
    ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
  )
  if found.returncode != 0:
    raise ValueError(f"{root} holds no history to read a revision from")
  return found.stdout.strip()


def model_sha() -> str:
  """What the gripper the demonstration drove is, as a capture names its URDF."""
  assets = Path(__file__).resolve().parent.parent / "assets"
  digest = hashlib.sha1()
  for asset in sorted(assets.glob("*.xml")):
    digest.update(asset.read_bytes())
  return digest.hexdigest()


def row_indexes(times: Float64Array, f_row: float) -> list[Any]:
  """The session clock at the instants a series was measured, and the row of
  the lattice each of them labels."""
  return [
    rr.TimeColumn(
      schema.TIMELINE,
      duration=np.rint(times * schema.NS_PER_SECOND).astype("timedelta64[ns]"),
    ),
    rr.TimeColumn(
      schema.ROW_TIMELINE, sequence=np.rint(times * f_row).astype(np.int64)
    ),
  ]


def frame_clock(
  times: Float64Array, fps: float, stride: int, f_row: float
) -> list[Any]:
  """A frame's own instant, the row its nominal time labels, and its keyframe."""
  rows = np.rint(times * fps).astype(np.int64)
  return [
    rr.TimeColumn(
      schema.TIMELINE,
      duration=np.rint(times * schema.NS_PER_SECOND).astype("timedelta64[ns]"),
    ),
    rr.TimeColumn(
      schema.ROW_TIMELINE, sequence=np.rint(rows / fps * f_row).astype(np.int64)
    ),
    rr.TimeColumn(
      schema.KEYFRAME_TIMELINE, sequence=schema.keyframe_index(rows, stride)
    ),
  ]


def hand_chunks(episode: Episode, hand: str, slot: str, size: int) -> list[Chunk]:
  """Everything the layer holds for one hand."""
  steps = np.asarray(episode.step_times, np.float64)
  poses = np.asarray(episode.poses[hand], np.float64)
  twists = np.asarray(episode.twists[hand], np.float64)
  frames = np.asarray(episode.frame_times, np.float64)
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
      indexes=frame_clock(frames, CAPTURE_FPS, IMAGE_STRIDE, episode.f_row),
      columns=rr.EncodedImage.columns(
        blob=blobs, media_type=["image/png"] * len(blobs)
      ),
    ),
    Chunk.from_columns(
      schema.state_pose(hand),
      indexes=row_indexes(steps, episode.f_row),
      columns=rr.Transform3D.columns(
        translation=poses[:, :3], quaternion=poses[:, [4, 5, 6, 3]]
      ),
    ),
    Chunk.from_columns(
      schema.state_lin_vel(hand),
      indexes=row_indexes(steps, episode.f_row),
      columns=rr.Scalars.columns(scalars=twists[:, :3]),
    ),
    Chunk.from_columns(
      schema.state_ang_vel(hand),
      indexes=row_indexes(steps, episode.f_row),
      columns=rr.Scalars.columns(scalars=twists[:, 3:]),
    ),
    Chunk.from_columns(
      schema.state_opening(hand),
      indexes=row_indexes(steps, episode.f_row),
      columns=rr.Scalars.columns(
        scalars=np.asarray(episode.openings[hand], np.float64)[:, None]
      ),
    ),
  ]


def cube_chunk(episode: Episode) -> Chunk:
  """Where the cube stood at every step, as a pose in the room frame."""
  poses = np.asarray(episode.cube, np.float64)
  return Chunk.from_columns(
    CUBE_POSE,
    indexes=row_indexes(np.asarray(episode.step_times, np.float64), episode.f_row),
    columns=rr.Transform3D.columns(
      translation=poses[:, :3], quaternion=poses[:, [4, 5, 6, 3]]
    ),
  )


def write_layer(
  path: Path,
  session: str,
  episode: Episode,
  size: int,
  seed: int,
  scale: int,
  started_at: int,
  sha: str,
) -> None:
  """Write one episode under its session id, in the layer's own shape."""
  chunks = [
    chunk
    for hand, slot in zip(schema.HANDS, SLOTS, strict=True)
    for chunk in hand_chunks(episode, hand, slot, size)
  ]
  chunks.append(cube_chunk(episode))
  written = datetime.now(timezone.utc).isoformat(timespec="seconds")
  account = {
    "config": path.parent.name,
    "f_step": episode.f_row,
    "started_at": started_at,
    "estimated_at": written,
    "written_at": written,
  }
  settings = {
    "scale": scale,
    "image_stride": IMAGE_STRIDE,
    "mask_radius": MASK_RADIUS,
    "side": SIDE,
    "cut_head": 0.0,
    "cut_tail": 0.0,
    "git_sha": sha,
    "estimate_git_sha": sha,
    "urdf_sha": model_sha(),
    "rerun_version": rr.__version__,
    # A capture names the solve it came from; a demonstration names the draw
    # that produced it, which is what reproduces the episode.
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
      rr.log(schema.STATE, values(**settings), static=True, recording=stream)
      send_chunks(store, recording=stream)
    partial.replace(path)
  finally:
    partial.unlink(missing_ok=True)
