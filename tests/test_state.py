"""An episode written as a state layer, read back as a trainer reads a capture's."""

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fixtures.scene import HOME, Bare
from specialist_sim import episode, rotation
from specialist_sim.camera.lens import Lens
from specialist_sim.env import ROWWISE, Cfg, Env, hold, make
from specialist_sim.episode import Episode
from specialist_sim.gripper import HANDS

pytest.importorskip("specialist_dataset", reason="the dataset extra is not installed")

from specialist_dataset import schema  # noqa: E402
from specialist_dataset.layer import encode  # noqa: E402
from specialist_dataset.read import Recording  # noqa: E402

from specialist_sim.scripts.state import exposures, main, twist  # noqa: E402

WIDTH = 16
LAYERS = {
  "right": ((), {"right": "cam00"}),
  "left": (("--hands", "left"), {"left": "cam00"}),
  "both": (("--hands", "both"), {"left": "cam00", "right": "cam01"}),
}


def flat(table: Any, entity: str, component: str) -> Any:
  """Every value of one entity's component, its rows end to end."""
  values = table.column(schema.selector(entity, component)).to_pylist()
  return np.concatenate([np.asarray(value).reshape(-1) for value in values])


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """A short glide, turn and grip of the bare room's hands, saved."""
  lenses = {hand: replace(Lens(), mask_radius=0.45) for hand in HANDS}
  env = Env(Bare(), Cfg(clock=ROWWISE, image_hands=(), lenses=lenses))
  env.reset([0])
  taken = [env.rows(0)]
  for k in range(60):
    target = HOME.copy()
    target[:, 0] += 0.002 * k
    turn = rotation.exp(np.array([0.0, 0.0, 0.01 * k]))
    target[:, 3:7] = rotation.multiply(turn, HOME[:, 3:7])
    target[:, 7] = 0.2 + 0.005 * k
    env.step(hold(target, 1)[None])
    taken.append(env.rows(0))
  one = Episode("one", episode.joined(taken), env.describe())
  return episode.save(tmp_path_factory.mktemp("episodes"), one)


@pytest.fixture(scope="module", params=LAYERS)
def layer(
  request: pytest.FixtureRequest,
  recorded: Path,
  tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, str]]:
  """A layer sim-state writes 16 px across, and its hands' slots."""
  options, slots = LAYERS[request.param]
  dataset = tmp_path_factory.mktemp(request.param)
  argv = [str(recorded), "--output-dir", str(dataset), "--image-size", str(WIDTH)]
  with pytest.MonkeyPatch.context() as patch:
    patch.setattr(sys, "argv", ["sim-state", *argv, "--supersample", "2", *options])
    main()
  return schema.layer_rrd(dataset, "state_s180_r100", "one"), slots


def test_lenses_fire_at_29_97_hz_on_shared_labels_inside_the_episode() -> None:
  times, label, capture = exposures(np.array([0.0, -0.45]), 300, 100.0)
  assert len(label) == 89 and label[0] == 3 and label[-1] == 297
  assert times.min() >= 0.0
  assert np.array_equal(capture, np.rint(times * 100))
  assert np.array_equal(capture[0], label), "cam00 sets the session clock"
  assert (capture[0] != capture[1]).any(), "each lens keeps its own shutter"
  assert np.allclose(np.diff(times), 1001 / 30000)
  _, label, capture = exposures(np.array([0.0, 0.4]), 281, 100.0)
  assert len(label) == 84 and label[-1] == 277 and capture.max() == 278


def test_twist_is_the_central_difference_one_sided_at_the_ends() -> None:
  t = np.arange(50)[:, None, None] / 100.0
  p0 = np.array([[0.4, 0.1, 0.0], [0.5, -0.2, 0.1]])
  v = np.array([[0.1, -0.2, 0.05], [-0.3, 0.0, 0.2]])
  a = np.array([[0.4, 0.0, -0.2], [0.0, 0.6, 0.1]])
  omega = np.array([[0.0, 0.0, 1.5], [0.4, -0.8, 0.3]])
  state = np.zeros((50, 2, 8))
  state[..., :3] = p0 + v * t + 0.5 * a * t**2
  state[..., 3:7] = rotation.multiply(rotation.exp(omega * t), HOME[:, 3:7])
  lin_vel, ang_vel = twist(state, 100.0)
  assert np.allclose(lin_vel, v + a * np.clip(t, 0.005, 0.485), atol=1e-9)
  assert np.allclose(ang_vel, np.broadcast_to(omega, ang_vel.shape), atol=1e-6)


def test_the_layer_holds_each_hand_as_a_capture_holds_its_camera(
  recorded: Path, layer: tuple[Path, dict[str, str]]
) -> None:
  out, slots = layer
  state = episode.load(recorded).rows.state
  lin_vel, ang_vel = twist(state, 100.0)
  with Recording(out) as r:
    account = {
      k: np.asarray(v).reshape(-1)[0] for k, v in r.property(schema.STATE).items()
    }
    want = {"config": "state_s180_r100", "f_row": 100.0, "scale": 180, "side": "rear"}
    want |= {"mask_radius": 0.5, "supersample": 2, "task": "fixtures.scene:Bare"}
    assert "revision" in account and account == account | want
    paths = set(r.entity_paths())
    hands = {path.split("/")[1] for path in paths if path.startswith("state/")}
    assert hands == set(slots) and "world" in paths
    for hand, slot in slots.items():
      i = HANDS.index(hand)
      assert r.row(schema.state_hand(hand))["slot"] == [slot]
      placed = r.row(schema.state_image(hand))[schema.POSE_TRANSLATION]
      assert np.asarray(placed).reshape(3).tolist() == [i * WIDTH, 0.0, 0.0]
      pose = schema.state_pose(hand)
      table = r.series(f"/{pose}")
      assert set(flat(table, pose, "Transform3D:child_frame")) == {f"{slot}_base"}
      assert set(flat(table, pose, "Transform3D:parent_frame")) == {"world"}
      rows = table.column(schema.ROW_TIMELINE).to_numpy()
      t, p = r.values(pose, schema.POSE_TRANSLATION, 3)
      assert np.array_equal(rows, np.arange(len(state))) and np.allclose(t, rows / 100)
      assert np.allclose(p, state[:, i, :3])
      q = r.values(pose, schema.POSE_QUATERNION, 4)[1]
      assert np.allclose(np.abs((q * state[:, i, [4, 5, 6, 3]]).sum(1)), 1.0)
      assert (q[:, 3] >= 0).all()
      for entity, values in (
        (schema.state_lin_vel(hand), lin_vel[:, i]),
        (schema.state_ang_vel(hand), ang_vel[:, i]),
        (schema.state_opening(hand), state[:, i, 7:]),
      ):
        assert np.allclose(r.scalars(entity, values.shape[1])[1], values)
      for joint, sign in (("left_finger", 1), ("right_finger", -1)):
        finger = schema.state_finger(hand, joint)
        turned = r.values(finger, schema.POSE_QUATERNION, 4)[1]
        assert np.allclose(turned[:, 2], np.sin(sign * state[:, i, 7] / 2), atol=1e-6)


def test_an_image_is_its_lens_traced_at_its_capture_row(
  recorded: Path, layer: tuple[Path, dict[str, str]]
) -> None:
  out, slots = layer
  one = episode.load(recorded)
  label = None
  with Recording(out) as r:
    for hand in slots:
      entity = schema.state_image(hand)
      table = r.series(f"/{entity}")
      capture = flat(table, entity, schema.CAPTURE_ROW).astype(np.int64)
      shutter = r.seconds(table)
      assert np.array_equal(capture, np.rint(shutter * 100))
      assert np.allclose(np.diff(shutter), 1001 / 30000)
      label = capture if label is None else label
      assert np.array_equal(table.column(schema.ROW_TIMELINE).to_numpy(), label)
      blobs = table.column(schema.selector(entity, "EncodedImage:blob"))
      blob = blobs.combine_chunks().drop_null().to_pylist()[0][0]
      lenses = {hand: Lens()}
      env = make(
        one.meta, image_size=WIDTH, supersample=2, image_hands=(hand,), lenses=lenses
      )
      traced = env.trace(one.rows.qpos[capture[:1]])[0][0, 0]
      assert bytes(blob) == encode(traced)
