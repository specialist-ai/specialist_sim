# specialist_sim

A simulated capture rig for collecting demonstrations and evaluating
manipulation policies: grippers on their embodiment, standing in whatever scene
an env puts them in. One env comes with it, pick-and-place.

## Setup

You need Python 3.12 and a CUDA GPU for the lenses, which Warp traces. A rig
with its lenses off runs on the CPU alone.

```bash
uv sync
uv run pytest
```

## The lenses

`Camera` carries the MEI intrinsics at the width they were measured at, the
lens's seat on the gripper base, and the rate it fires at. `NOMINAL` is the
rig's own. A lens keeps the disc inscribed in its frame and is black outside
it, as a recorded image does. `Cfg.image_size` is how many pixels across a
frame is traced, 360 by default, which is the width a trainer crops from.
`render.Renderer` draws any camera on any body.

## The rig

```python
import mujoco
from specialist_sim.rig import Cfg, Rig
from specialist_sim.action import hold

rig = Rig(Cfg(scene=mujoco.MjSpec))     # the hands in an empty scene
rig.place(0, hands)                     # hands (2, 8): position, quaternion, finger angle
observation = rig.begin([0])
observation = rig.step(hold(observation.state[:, -1]))
```

A hand is eight numbers: its base tag's room-frame position, a rotation as a
`wxyz` quaternion, and its left finger's angle in radians. Every array has a
leading axis over the `num_envs` parallel envs the rig holds.

| field | shape | meaning |
| --- | --- | --- |
| `observation.row` | `(num_envs, rows)` int64 | the rows this step spanned |
| `observation.state` | `(num_envs, rows, driven, 8)` | each driven hand at each row |
| `observation.image` | `(num_envs, lenses, size, size, 3)` uint8 | each lens's newest frame, black outside the disc |
| `observation.body` | `(num_envs, lenses, size, size)` int32 | the body each pixel struck, `-1` for none |
| `observation.image_row` | `(num_envs, lenses)` int64 | the row each frame was taken at |
| `action` | `(num_envs, chunk_size, driven, 8)` | the action chunk the driven hands follow |

`Cfg.scene` is whatever the hands stand in, as a call that returns the
`MjSpec` they are attached into. `Cfg.driven` names the hands the action
commands. Every other hand holds where it was last sent. `Cfg.camera_latency`
is the seconds a frame takes to reach a policy once the shutter has fallen,
the sensor's own delay and nothing else. A free-running lens carries the frame
in hand up to one frame period past it.

The envs' data are the truth between steps: anything moved on
`rig.datas[env_idx]` is what the next step runs from, and an env puts its own
things there between `place` and `begin`. Each env steps its own `MjData`
with `mj_step` on a thread pool, which MuJoCo runs without the GIL, so an env
lands exactly where it would alone and keeps its solver's warm start from one
physics step to the next.

After a step the rig reports `unstable`, whether MuJoCo reset an env it could
not carry, and `tracking_error`, the largest gap between a driven hand and its
setpoint. Every row's joint coordinates sit in `rig.qpos_rows` and
`rig.qvel_rows`, and its hands in `rig.state_rows`.

### Time

Time is counted in rows, the ticks of the lattice a dataset records a hand at.
`Clock` fixes every rate as a whole number of the next finer one:

| field | default | meaning |
| --- | --- | --- |
| `physics_dt` | 1/1200 s | one physics step |
| `per_row` | 12 | physics steps in a row: 100 Hz rows |
| `per_action` | 2 | rows in one action of a chunk: 50 Hz actions |
| `per_step` | 10 | rows in a control step: 10 Hz steps |

A control step executes the `n_action_steps` actions that fall inside it,
five by default, and discards the rest. A policy predicts a chunk of
`CHUNK_SIZE`, sixteen, so it commits to more motion than one step executes and
resolves it finer than one step could. The setpoint walks the interpolated path
at the physics rate with the velocity of the segment it is on, and the
embodiment chases it every physics step. `action.hold` builds the action chunk
that asks to stay put.

Nothing synchronizes a lens to any of this. Each fires at its own rate on the
physics lattice, from wherever in its period an episode opens it, so it slips
against the control step and against the other lens, and its shutter falls
between rows as a capture's does. An episode draws where each lens stands from
the seed it is reproduced by. The observation carries each lens's newest
exposure older than `camera_latency`, beside the row it was taken at. Only
the exposures an observation delivers are traced.

### Choosing the embodiment

`Cfg.embodiment` names what holds the grippers up and which slots,
`left_gripper` and `right_gripper`, the scene holds. Another embodiment
implements `specialist_sim.embodiment.Embodiment`: it stands the hands in the
scene, binds their ids, puts a hand at a pose, and writes `ctrl` and `xfrc`
for every env each physics step. An embodiment's `attach` stands every body of
a hand under its slot's name. A hand's state is its base tag's pose whatever
carries it, so a policy is not told. `tests/gantry.py` is such an
embodiment, a joint chain driven through `ctrl`, and the seam tests run on it
beside the free rig.

The default embodiment, `Floating`, flies each gripper on a massless free
joint: an impedance of 4000 N/m and 40 N m/rad pulls it toward its setpoint,
the segment's velocity fed forward and the gripper's weight cancelled. Its pull
is capped at 25 N and 1.5 N m, so a hand stops when the fingertips meet a
surface instead of driving through it. The cap bounds a push, not the speed: a
hand follows its setpoint at up to 3 m/s and 20 rad/s, and closes a gap to it
at no more than 1.5 m/s and 10 rad/s.

### Choosing the cameras

Every lens is configuration, defaulting to the nominal camera:

```python
from specialist_sim.camera import Camera

Cfg(scene=..., cameras={"right_gripper": measured})
```

## The env and its task

```python
from specialist_sim.env import Cfg, Env
from specialist_sim.pickplace import PickPlace

env = Env(PickPlace(), Cfg(driven=("right_gripper",)), num_envs=8)
observation, report = env.reset(range(8))
```

One `Env` stands every scene. It owns the rig and the latches every task
shares: a goal held for `Task.dwell` steps latches `success`, which pays one
`reward`, and `done` is success, the task's own loss, or an env MuJoCo could
not carry. A label the task names in `Task.latched` is held once it is true, so
its rate reads as "this happened", not "this was true at the end".

A `Task` says what the rest is. `spec` builds the scene, `bind` reads what
judging needs out of the compiled model, `open` lays one env's episode from its
seed and says where each lens stands in its period, and `judge` returns a
`Verdict`: how far every env has come, whether its goal holds, whether it is
lost, and whatever else the task counts under its own names. `Task.name` is
what an episode calls the scene, and `Task.focus` the room point an overview of
it looks at. `env.task` is the task itself, so a scene's own reading is reached
through it.

A task's knobs are the task's own, `PickPlace(region=...)`, and `Cfg` carries
only what any scene runs under. A task names the `Cfg` it expects by default,
which `Env` uses when it is given none.

## Physics

Every solver and contact setting is one value, `physics.Physics`, which the rig
applies to every scene. The values come from MuJoCo's documentation (solver
parameters, slip, integrators) and from what MuJoCo Menagerie's grippers use:
an `implicitfast` integrator, an elliptic friction cone at an impedance ratio
of 100, a 2 ms contact time constant with the gripping pads taking priority,
and object damping near zero.

## Evaluation

`eval.eval_policy` runs one episode per seed on a `Policy` through an `Env`
standing any task, keeping every env busy: an env whose episode ends restarts
alone on the next seed. A task counts whatever it likes under its own names,
in `report.labels`, and an outcome carries them. Evaluation rates each one
without knowing what any of them mean. `eval.summary` reports the success rate
and those rates beside everything they were produced under: the task and its
knobs, the configuration, and a `revision` that hashes the gripper, the task,
what it ran under, the physics engine and the files under the task's `assets`.

## The pick-and-place task

```python
from specialist_sim.env import Env
from specialist_sim.pickplace import PickPlace
from specialist_sim.action import hold

env = Env(PickPlace())
observation, report = env.reset([0])
observation, report = env.step(hold(observation.state[:, -1]))
```

Two hands stand at the desk with a tape target on its middle and a cube. The
cube starts clear of the tape, within `PickPlace.region` of it, and the episode
is won when it rests on the tape, let go of.

`report.labels` carries what the task counts of every env's cube (`on_tape`,
`held`, `dropped`, `picked`) beside how the episode stands (`progress`,
`success`, `unstable`, `tracking_error`, `reward`, `done`). The reward is one
on the step the cube has settled on the tape for five steps, so an undiscounted
return is the success indicator.

`Cfg.driven` names the hands a policy commands, both or one of them, and
`Cfg.seen` any lens the observation carries beyond their own. A start of your
own repeats an episode exactly:

```python
env.reset([0], start=(hands, cube))   # hands (2, 8); cube (x, y, yaw)
```

## Layout

```
src/specialist_sim/
  env.py            the Env every task is stood in, and the Task interface
  rig.py            the hands on their embodiment in num_envs envs, and the Clock
  gripper.py        the described gripper, its base, its pads
  embodiment.py     what holds the hands up: Floating
  action.py         action chunks, and the Interpolator the embodiment chases
  camera.py         the MEI fisheye, which render.py and kernels.py trace on the GPU
  eval.py           Policy, eval_policy, summary, mcnemar
  pickplace/        the pick-and-place task: room, task
tests/
```
