"""What the solver and contact settings have to deliver. Each threshold sits
between what the rig's physics measures and what a wrong setting measures, so a
regression trips it and ordinary noise does not."""

import mujoco
import numpy as np

from fixtures.lego import brick_geoms, brick_q
from specialist_sim import rotation
from specialist_sim.arrays import BoolArray, Float64Array
from specialist_sim.env import ROWWISE, Cfg, Env, Start
from specialist_sim.gripper import GRIPPING, HANDS
from specialist_sim.tasks.sort_lego import SortLego, initial
from specialist_sim.tasks.sort_lego.measures import in_bin

# No lens, and one commanded pose a row: the plant alone, along a path.
PLANT = Cfg(clock=ROWWISE, image_hands=())
ROW_HZ = 1.0 / PLANT.clock.row_dt


def _env() -> Env[SortLego]:
  return Env(SortLego(mode="pile", bricks=1), PLANT)


def _deepest(data: mujoco.MjData, first: BoolArray, second: BoolArray) -> float:
  """The deepest any geom of ``first`` sinks into one of ``second``."""
  deepest = 0.0
  for contact in data.contact:
    one, other = contact.geom
    if (first[one] and second[other]) or (first[other] and second[one]):
      deepest = max(deepest, -contact.dist)
  return deepest


def _grasp(brick: Float64Array, hand: str) -> Float64Array:
  """The hand state that holds ``brick`` between its pads, fingers along its
  long side and pointing away from the hand's home."""
  long_side = rotation.rotate(brick[3:7], np.array([1.0, 0.0, 0.0]))
  along = np.array([long_side[0], long_side[1], 0.0]) / np.linalg.norm(long_side[:2])
  if np.dot(along[:2], brick[:2] - initial.home(hand)[:2]) < 0.0:
    along = -along
  pitch = np.radians(48.0)  # the fingers tip down this far to reach a flat brick
  turn = rotation.pointing(np.cos(pitch) * along - np.array([0.0, 0.0, np.sin(pitch)]))
  # A held brick's middle in the base tag's frame: z at the pads' centers, y mid-jaw.
  at = brick[:3] - rotation.rotate(turn, np.array([0.098, 0.0, -0.0177]))
  return np.array([*at, *turn, 0.3])  # open just wide of the brick


def _eased(plan: list[tuple[float, Float64Array]], at: float) -> Float64Array:
  """A hand state along ``plan``, smoothstepped between its arrival times."""
  if at <= plan[0][0]:
    return plan[0][1]
  for (first, a), (second, b) in zip(plan, plan[1:], strict=False):
    if at <= second:
      part = (at - first) / (second - first)
      eased = part * part * (3.0 - 2.0 * part)
      out = a + eased * (b - a)
      out[:7] = rotation.interpolate(a, b, eased)
      return out
  return plan[-1][1]


def test_a_shaken_grasp_neither_slips_nor_sinks() -> None:
  """The elliptic cone with an impedance ratio of 100 holds a brick through a
  shake, and the 2 ms contact time constant keeps the pads off its surface. A
  ratio of 10 slips 2.0 mm; softer pads sink 7.0 mm."""
  env = _env()
  bricks = brick_geoms(env)
  pads = env.model.geom_condim == GRIPPING
  observation, _ = env.reset([0])
  which = int(np.flatnonzero(in_bin(env.objects().laid[0, :, :3]) < 0)[0])
  homes = observation.state[0, -1].copy()
  grasp = _grasp(env.objects().poses[0, which], HANDS[1])
  above = grasp.copy()
  shut = grasp.copy()
  above[2] += 0.06
  shut[7] = -0.15  # the commanded squeeze on the brick
  lifted = shut.copy()
  lifted[2] += 0.10
  plan = [(0.0, homes[1]), (0.8, above), (1.3, grasp), (1.7, shut), (2.6, lifted)]
  data = mujoco.MjData(env.model)
  sinks: list[float] = []
  held: Float64Array | None = None
  for row in range(round(4.8 * ROW_HZ)):
    at = row / ROW_HZ
    target = homes.copy()
    target[1] = _eased(plan, at)
    if at >= 2.8:
      phase = 2.0 * np.pi * 1.5 * (at - 2.8)
      ramp = min(1.0, (at - 2.8) / 0.3)
      target[1, :3] += 0.05 * np.sin(phase + np.array([0.0, 2.1, 4.2])) * ramp
    observation, _ = env.step(target[None, None])
    if at >= 2.0:
      sinks.append(_deepest(env.sim.get_data(0, data), pads, bricks))
    if held is None and at >= 2.8:
      held = rotation.compose(
        rotation.inverse(observation.state[0, -1, 1]), env.objects().poses[0, which]
      )
  assert held is not None
  end = rotation.compose(
    rotation.inverse(observation.state[0, -1, 1]), env.objects().poses[0, which]
  )
  slip = float(np.linalg.norm(end[:3] - held[:3]))
  assert slip < 1.5e-3, f"the brick slipped {1e3 * slip:.1f} mm in the grasp"
  sank = float(np.median(sinks))
  assert sank < 2e-3, f"the pads sank {1e3 * sank:.1f} mm into the brick"


def test_resting_bricks_stop_moving() -> None:
  """A 2 ms contact time constant lets a pile come down and stay down."""
  env = _env()
  bricks = brick_geoms(env)
  rng = np.random.default_rng(0)
  pile = initial.draw("pile", rng, 8)
  env.reset([0], start=Start(initial.starts(HANDS), pile))
  for _ in range(round(1.9 * ROW_HZ)):
    env.step()
  before = env.objects().poses[0]
  for _ in range(round(ROW_HZ)):
    env.step()
  after = env.objects().poses[0]
  crept = float(np.linalg.norm(after[:, :3] - before[:, :3], axis=1).max())
  assert crept < 2e-3, f"a resting brick crept {1e3 * crept:.1f} mm in a second"
  sank = _deepest(env.sim.get_data(0), bricks, bricks)
  assert sank < 2e-3, f"bricks sank {1e3 * sank:.1f} mm into each other"


def test_a_brick_in_flight_keeps_its_spin() -> None:
  """Flight takes almost nothing out of a tossed brick's spin; the footage turns
  at about 25 rad/s."""
  env = _env()
  env.reset([0])
  q = brick_q(env, 0)
  at = env.model.joint("brick_0_joint").dofadr[0]
  env.sim.qpos[0, q : q + 7] = [0.3, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0]
  spinning = np.array([25.0, 5.0, 3.0])
  env.sim.qvel[0, at : at + 6] = [0.0, 0.0, 0.0, *spinning]
  for _ in range(round(0.05 * ROW_HZ)):
    env.step()
  left = float(
    np.linalg.norm(env.sim.qvel[0, at + 3 : at + 6]) / np.linalg.norm(spinning)
  )
  assert left > 0.5, f"only {100 * left:.0f}% of the spin survived 50 ms of flight"
