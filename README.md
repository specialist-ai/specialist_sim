# specialist_sim

A simulated capture rig for evaluating manipulation policies. Two flying
grippers pick and place a cube on a table, each carrying a fisheye camera that
renders through the same lens model the real captures were solved with.

```python
from specialist_sim import PickPlaceEnv
from specialist_sim.action import hold

env = PickPlaceEnv()
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(hold(observation["state"]))
```

The interface follows Gymnasium; the package does not depend on it.

## Observation and action

Both hands appear in one flat vector, eight numbers each: a room-frame
position, a rotation as a quaternion, and a finger opening from shut to the
joint's full travel. One such vector is a pose.

| key | shape | meaning |
| --- | --- | --- |
| `observation["left_gripper"]`, `["right_gripper"]` | `(image_size, image_size, 3)` uint8 | what each gripper's lens sees, blanked outside the disc |
| `observation["state"]` | `(16,)` float32 | both hands now, in one pose's layout |
| `action` | `(16, 16)` float64 | the trajectory both hands follow over the next 0.32 s |

Because the layouts match, a residual policy is a subtraction. The reward is
one on the step the cube settles onto the tape unheld, so an undiscounted
return is the success indicator.

A lens keeps the middle `MASK_RADIUS` of its width and is black outside it,
which is the disc a recorded image keeps. The projection stays invertible
a little past that, so the mask, not the lens model, is what bounds the image.
The camera watching the room keeps its whole frame.

### The action is a trajectory

An action names 16 poses 1/50 s apart. A control step runs at 10 Hz, so it
tracks the leading five and discards the other eleven: the policy commits to
0.32 s of motion, over three times what one step executes, and resolves that
motion five times finer than one step could. `action.hold(state)` builds the
trajectory that asks to stay put.

The tracker underneath is a stiff impedance on each free-flying gripper. It
resamples the trajectory onto the physics step, so the setpoint walks the
commanded path at 2 kHz rather than jumping once per control step, and its
velocity feedforward comes from the leg it is on.

The three rates are locked together: a control step must span a whole number of
poses, and a pose a whole number of physics steps. At 2 kHz physics and a 50 Hz
trajectory, the control rate has to divide 50. The horizon is `HORIZON / 50` s,
so it moves in 20 ms steps: raising `HORIZON` alone changes nothing else.

`env.observation_space` and `env.action_space` describe these as Gymnasium
objects, from the environment's own configuration. They need the `gym` extra.

`env.render()` returns a view of the room from a fixed camera when the
environment is built with `render_mode="rgb_array"`. Alongside it,
`env.views[slot].body` names the body each pixel struck.

## Metrics

The numbers a policy is judged on are defined here, in the package that owns the
environment they are statements about, so that two people reporting a success
rate are reporting the same quantity:

```python
from specialist_sim.metrics import Episode, summarise

scoring = Episode(env)
observation, info = env.reset(seeds=seeds)
scoring.begin(seeds=seeds)
while scoring.running.any():
  action = policy(observation)
  observation, reward, terminated, truncated, info = env.step(action)
  scoring.record(action, observation, reward, terminated, truncated, info)
scoring.stop()          # for anything the caller's own budget ended
print(summarise(scoring.reports()))
```

Nothing here knows what a policy is: it is handed what a caller already has, the
action it commanded and the five things `step` returned. There is one accumulator
per world, because the environment is batched and an episode's numbers belong to
the world that ran it — a world that ends early keeps its report while the rest
run on.

| | |
| --- | --- |
| `success` | the task latched: the cube back on the tape, at rest and let go of, having been off it |
| `picked` | a success the hand *carried* — the cube came clear of the table and was between the pads. A nudge across the tape is not |
| `dropped` | the cube off the table |
| `stage`, `progress` | how far along `STAGES` — reached, touched, lifted, carried, placed — the episode got, and that as a fraction |
| `stages` | which of them actually happened, which is not the same question |
| `lift`, `closest_approach`, `held_steps`, `carried_steps`, `first_contact` | how far the cube got, how near a grasp point came to it, how long a pad was on it, and when one first was |
| `path_length`, `smoothness` | how far the hands travelled and how fluently, as a spectral arc length |
| `clipped` | the fraction of commanded action components that fell outside the action box |
| `tracking`, `tracking_rotation` | how far the worst hand ended from the pose the step actually executed — the rig's fidelity, not the policy's |
| `image_age`, `penetration` | the stalest frame delivered, and the deepest two bodies overlapped |

### What the numbers are careful about

**Everything comes with an interval.** A success rate over twenty episodes is
worth about plus or minus twenty points, so two policies four points apart over
twenty episodes have not been told apart, and reporting them as though they had
is how an evaluation accumulates gains that were never there. Rates get a Wilson
score interval — tighter than the exact Clopper–Pearson one at these sample
sizes, and sensible at 0 and 1 where the normal approximation claims certainty
from nothing. Distributions get a seeded percentile bootstrap. Conditional means
carry the `_n` they were taken over.

**A binary outcome throws away most of an episode.** A policy frozen in place and
one that drove a hand onto the cube and failed to close both score zero.
`progress` grades them apart, which is also what lets a comparison need far fewer
episodes: most of them stop being ties.

**The stage rates are the failure breakdown, and they need not fall.** Each is
the share of episodes in which that thing happened, not the share that got at
least that far, because the stages are not strictly ordered here — the back of a
gripper can knock the cube into the air, which is `lifted` with no pad ever on
it. Where the sequence rises, that is the finding: a policy scattering the cube
rather than picking it up. Forcing it to fall would report grasps that never
happened.

**Two policies can succeed alike and move nothing alike.** `smoothness` is the
spectral arc length of the hands' speed: −1.40 for a single minimum-jerk reach,
falling with every extra submovement, and unmoved by noise. Jerk was the
alternative and is worse here — a third derivative of sampled data, and every
dimensionless form of it wants a movement that starts and ends at rest, which
continuous pick and place does not.

**`clipped` is a fraction, not a count**, so it means the same thing over a run
of any length and on a rig with any number of hands.

**One episode may not be counted twice.** `summarise` refuses a set that repeats
a seed, which is easy to do by accident when a batch of worlds is reset from one
seed and makes every rate wrong when it happens.

**These intervals are about this simulator.** They describe sampling error over
the episodes drawn here. The gap between this rig and a real one is a bias, and
no number of simulated episodes shrinks it — predicting hardware from simulation
needs paired real trials, not more simulated ones.

`summarise` deliberately does not test whether two runs differ. Comparing
policies wants a paired test over the per-episode reports, and the reports are
kept so a caller can run one. The module docstrings of `metrics.episode` and
`metrics.summary` are the specification; `metrics.confidence` and
`metrics.smoothness` are separate modules and either can be deleted without
touching the rest.

## Choosing the cameras

Every lens is configuration, defaulting to a nominal camera:

```python
from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.calibration import CameraCfg

env = PickPlaceEnv(PickPlaceCfg(cameras={"left_gripper": CameraCfg(), "right_gripper": measured}))
```

`CameraCfg` carries the MEI intrinsics at the width they were measured at and
the lens's seat on the gripper base, so a caller holding a real calibration
renders through it instead.

## Starting where you choose

`reset` takes a state and a cube position, for an evaluation that repeats one
episode exactly:

```python
env.reset(seed=0, options={"state": start, "cube": (0.52, 0.06)})
```
