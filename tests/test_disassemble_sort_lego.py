"""The LEGO disassembly and sorting room: how pairs are laid, when a bond breaks,
how a tray is judged, and a hand pulling a pair apart."""

from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
import pytest

from fixtures.lego import eased, in_hand, lift
from fixtures.scene import BLIND, PLANT, ROW_HZ
from specialist_sim import rotation
from specialist_sim.arrays import Float64Array
from specialist_sim.env import Bond, Env, Objects, Start
from specialist_sim.gripper import HANDS
from specialist_sim.tasks import desk
from specialist_sim.tasks.disassemble_sort_lego import (
  BONDS,
  CLUTCH,
  FACE,
  PAIRS,
  PULL,
  SPACING,
  DisassembleSortLego,
  lay,
)
from specialist_sim.tasks.sort_lego import initial, measures
from specialist_sim.tasks.sort_lego.measures import BRICK_COLORS, in_bin

RED, BLUE = PAIRS[0]


def _stair(base: int, part: int) -> Float64Array:
  """Every brick's pose, ``part`` clutched on ``base`` flat on the tray and the
  rest parked."""
  poses = initial.waiting()
  poses[base] = [*measures.TRAY_CENTER, initial.LAID, 1.0, 0.0, 0.0, 0.0]
  poses[part] = rotation.compose(poses[base], np.asarray(CLUTCH))
  return poses


def _clutched(poses: Float64Array, base: int, part: int, within: float = 1e-5) -> bool:
  """Whether ``part`` lies at the clutch pose on ``base``, to ``within`` meters
  and radians."""
  on = rotation.compose(rotation.inverse(poses[base]), poses[part])
  turn = rotation.log(
    rotation.multiply(rotation.conjugate(np.asarray(CLUTCH[3:])), on[3:])
  )
  return bool(
    np.linalg.norm(on[:3] - CLUTCH[:3]) < within and np.linalg.norm(turn) < within
  )


def test_a_seed_lays_whole_clutched_pairs_spaced_on_the_tray_and_parks_the_rest() -> (
  None
):
  parked = initial.waiting()
  reds, blues = np.array(PAIRS).T
  counts = set()
  for seed in range(100):
    poses = lay(np.random.default_rng(seed), None)
    on = in_bin(poses[:, :3]) < 0
    assert np.array_equal(on, (poses != parked).any(axis=1))
    assert np.array_equal(on[reds], on[blues]), "a pair starts whole"
    counts.add(int(on[reds].sum()))
    middles = []
    for red, blue in np.array(PAIRS)[on[reds]]:
      assert _clutched(poses, blue, red) != _clutched(poses, red, blue)
      inside = np.asarray(measures.TRAY_SIZE) / 2 - measures.TRAY_RIM
      assert (np.abs(poses[[red, blue], :2] - measures.TRAY_CENTER) < inside).all()
      middles.append((poses[red, :2] + poses[blue, :2]) / 2)
    for k, middle in enumerate(middles):
      for other in middles[:k]:
        assert np.linalg.norm(middle - other) >= SPACING - 1e-9
  assert counts == {1, 2, 3}
  rng = np.random.default_rng
  assert np.array_equal(lay(rng(5), 2), lay(rng(5), 2))
  assert (in_bin(lay(rng(3), 5)[:, :3]) < 0).sum() == 10


def test_a_reset_settles_every_pair_still_clutched_and_none_apart() -> None:
  env = Env(DisassembleSortLego(pairs=3), BLIND, 4)
  _, info = env.reset(range(4))
  objects = env.objects()
  assert objects.bonded.sum(axis=1).tolist() == [3] * 4
  counted = {"pairs": 3, "apart": 0, "started": 6, "sorted": 0, "on_tray": 6}
  for key, count in counted.items():
    assert info["counts"][key].tolist() == [count] * 4, key
  assert not info["labels"]["apart"].any() and not info["done"].any()
  assert np.abs(objects.velocity).max() < 0.05, "a pair is still tumbling"
  for poses, laid in zip(objects.poses, objects.laid, strict=True):
    for red, blue in PAIRS:
      if in_bin(laid[red, :3]) < 0:
        on = _clutched(poses, blue, red, 2e-5) or _clutched(poses, red, blue, 2e-5)
        assert on, "a bond broke"
  for _ in range(10):
    env.step()
  assert env.objects().bonded.sum(axis=1).tolist() == [3] * 4


def _strain(env: Env[Any], k: int, force: Float64Array, torque: Float64Array) -> None:
  """Apply ``force`` and ``torque`` to bond ``k``'s part in env 0, about its
  anchor in its base's frame, and the opposite to its base."""
  bond = env.task.bonds[k]
  base, part = (env.model.body(name).id for name in (bond.base, bond.part))
  sim = env.sim
  to_room = rotation.to_matrix(sim.xquat[0, base])
  anchor = sim.xpos[0, base] + to_room @ bond.anchor
  pulled, turned = to_room @ force, to_room @ torque
  xfrc = sim.bind("xfrc_applied")
  xfrc[0, part] = [*pulled, *(turned - np.cross(sim.xipos[0, part] - anchor, pulled))]
  xfrc[0, base] = [*-pulled, *(np.cross(sim.xipos[0, base] - anchor, pulled) - turned)]


hx, hy = FACE


@pytest.mark.parametrize(
  ("force", "torque", "parts", "on_tray"),
  [
    pytest.param((0, 0, 0.95), (0, 0, 0), False, False, id="pulled_short"),
    pytest.param((0, 0, 1.05), (0, 0, 0), True, False, id="pulled"),
    pytest.param((0, 0, 0), (0, 0.95 * hx, 0), False, False, id="pried_short"),
    pytest.param((0, 0, 0), (0, -1.05 * hx, 0), True, False, id="pried_across"),
    pytest.param((0, 0, 0), (1.05 * hy, 0, 0), True, False, id="pried_along"),
    pytest.param((0, 0, 0.5), (0, 0.55 * hx, 0), True, False, id="pulled_and_pried"),
    pytest.param((0, 0, -2), (0, hx, 0), False, True, id="pushed_on_it"),
    pytest.param((0, 0, -0.6), (0, 1.8 * hx, 0), True, False, id="pushed_beyond_it"),
    pytest.param((0.9, -0.9, 0), (0, 0, 0), False, False, id="sheared"),
    pytest.param((0, 0, 0), (0, 0, 1.2 * hx), False, False, id="twisted"),
  ],
)
def test_a_bond_breaks_once_its_load_reaches_its_pull(
  force: tuple[float, ...], torque: tuple[float, ...], parts: bool, on_tray: bool
) -> None:
  """A bond holds against a push anywhere on its face and against any shear
  or twist. It breaks once its load reaches the pull its episode drew, with the
  part pulled off or pried up about an edge of the face. ``force`` and
  ``torque`` are in units of that pull. The base takes the opposite load, or the
  tray holds it."""
  env = Env(DisassembleSortLego(pairs=1), PLANT)
  env.reset([0], start=Start(initial.starts(HANDS), _stair(BLUE, RED)))
  (k,) = np.flatnonzero(env.objects().bonded[0]).tolist()
  pull = env.pulls[0, k]
  for _ in range(round(0.3 * ROW_HZ)):
    _strain(env, k, pull * np.asarray(force, float), pull * np.asarray(torque, float))
    if on_tray:
      env.sim.bind("xfrc_applied")[0, env.model.body(f"brick_{BLUE}").id] = 0.0
    env.step()
    if not env.objects().bonded[0, k]:
      break
  assert env.objects().bonded[0, k] != parts


def test_each_episode_draws_its_pulls_from_the_seed() -> None:
  """Each episode draws every bond's pull between the bounds in ``PULL``. The
  same seed draws the same pulls, and a reset redraws only the envs it opens."""
  env = Env(DisassembleSortLego(pairs=3), BLIND, 4)
  env.reset(range(4))
  pulls = env.pulls
  assert ((PULL[0] <= pulls) & (pulls <= PULL[1])).all()
  assert len(np.unique(pulls)) == pulls.size, "every bond draws its own pull"
  again = Env(DisassembleSortLego(pairs=3), BLIND, 4)
  again.reset(range(4))
  assert np.array_equal(again.pulls, pulls)
  env.reset([10], env_ids=[1])
  assert np.array_equal(np.delete(env.pulls, 1, axis=0), np.delete(pulls, 1, axis=0))
  assert not np.array_equal(env.pulls[1], pulls[1])


def test_a_pair_counts_apart_once_it_parts_and_each_brick_sorts_on_its_own() -> None:
  """A pair counts as apart once its bond breaks, and a brick still clutched is
  never sorted."""
  n = len(BRICK_COLORS)
  laid = _stair(BLUE, RED)[None]
  red_bin, blue_bin = (
    [*measures.bin_middle(color), desk.TOP + measures.BIN_FLOOR[2] + 0.01]
    for color in ("red", "blue")
  )
  held = np.zeros((1, len(BONDS)), bool)
  held[0, next(k for k, bond in enumerate(BONDS) if bond.part == f"brick_{RED}")] = True
  apart = np.zeros_like(held)

  def judged(red: list[float] | None, blue: list[float] | None, bonded: Any) -> Any:
    poses = laid.copy()
    for brick, at in ((RED, red), (BLUE, blue)):
      if at is not None:
        poses[0, brick, :3] = at
    still, free, ok = np.zeros((1, n, 6)), np.zeros((1, n), bool), np.zeros(1, bool)
    return DisassembleSortLego().judge(Objects(poses, still, free, laid, ok, bonded))

  verdict = judged(None, None, held)
  assert verdict.progress[0] == 0.0 and not verdict.labels["apart"][0]
  assert verdict.counts["pairs"][0] == 1 and verdict.counts["apart"][0] == 0
  verdict = judged(red_bin, None, apart)
  assert verdict.labels["apart"][0] and verdict.counts["sorted"][0] == 1
  assert verdict.progress[0] == pytest.approx(2 / 3)
  verdict = judged(red_bin, blue_bin, apart)
  assert verdict.progress[0] == 1.0 and not verdict.failed[0]
  verdict = judged(red_bin, None, held)
  assert verdict.counts["sorted"][0] == 0, "a clutched brick is not sorted"
  verdict = judged(red_bin, red_bin, held)
  assert verdict.failed[0], "the blue brick is in the red bin"


@pytest.mark.parametrize("stronger", [1.0, 2.0])
def test_a_grasp_pulls_a_brick_off_its_pair_and_keeps_it(stronger: float) -> None:
  """One hand lifts the top brick of a pair while the other brick is held down.
  The lift breaks the bond as it starts and carries the brick off without
  slipping, at the strongest pull an episode draws and at twice that pull."""

  @dataclass(frozen=True)
  class Clutched(DisassembleSortLego):
    bonds: ClassVar[tuple[Bond, ...]] = tuple(
      replace(bond, pull=(stronger * PULL[1], stronger * PULL[1])) for bond in BONDS
    )

  env = Env(Clutched(pairs=1), PLANT)
  laid = _stair(BLUE, RED)
  observation, _ = env.reset([0], start=Start(initial.starts(HANDS), laid))
  homes = observation.state[0, -1].copy()
  above, around, shut, lifted = lift(laid[RED], HANDS[1])
  plan = [(0.0, homes[1]), (0.8, above), (1.3, around), (1.7, shut), (2.0, shut)]
  plan.append((4.0, lifted))
  xfrc = env.sim.bind("xfrc_applied")
  base = env.model.body(f"brick_{BLUE}").id
  taken: Float64Array | None = None
  parted: float | None = None
  for row in range(round(4.5 * ROW_HZ)):
    at = row / ROW_HZ
    target = homes.copy()
    target[1] = eased(plan, at)
    xfrc[0, base, 2] = -4.0 * stronger * PULL[1]
    observation, _ = env.step(target[None, None])
    poses = env.objects().poses[0]
    if taken is None and at >= 1.95:
      taken = in_hand(observation.state[0, -1, 1], poses[RED])
    if parted is None and not env.objects().bonded[0].any():
      parted = at
  assert taken is not None and parted is not None
  assert 2.0 < parted < 2.3, f"the bond broke at {parted:.2f} s"
  end = in_hand(observation.state[0, -1, 1], poses[RED])
  slip = float(np.linalg.norm(end[:3] - taken[:3]))
  assert slip < 1.5e-3, f"the brick slipped {1e3 * slip:.1f} mm in the grasp"
  assert poses[RED, 2] - laid[RED, 2] > 0.09
  assert np.linalg.norm(poses[BLUE, :3] - laid[BLUE, :3]) < 2e-3
