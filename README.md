# specialist_sim

A simulated capture rig for evaluating manipulation policies: grippers on
their embodiment, standing in whatever scene an env puts them in. What holds a
gripper up is configuration: by default each flies free, pulled about by a
wrench.

## The lenses

`Camera` carries the MEI intrinsics at the width they were measured at, the
lens's seat on the gripper base, and the rate it fires at; `NOMINAL` is the
rig's own. A lens keeps the disc inscribed in its frame and is black outside
it, as a recorded image does. `render.Renderer` draws any camera on any body.

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

A hand is eight numbers: its base tag's room-frame position, a rotation
as a `wxyz` quaternion, and its left finger's angle in radians. Every array
has a leading axis over the `num_envs` parallel envs the rig holds.

| field | shape | meaning |
| --- | --- | --- |
| `observation.row` | `(num_envs, rows)` int64 | the rows this step spanned |
| `observation.state` | `(num_envs, rows, driven, 8)` | each driven hand at each row |
| `observation.image` | `(num_envs, lenses, size, size, 3)` uint8 | what each lens saw, black outside the disc |
| `observation.body` | `(num_envs, lenses, size, size)` int32 | the body each pixel struck, `-1` for none |
| `observation.image_row` | `(num_envs, lenses)` int64 | the row each lens's frame was taken at |
| `action` | `(num_envs, chunk_size, driven, 8)` | the action chunk the driven hands follow |

`Cfg.scene` is whatever the hands stand in, as a call that returns the
`MjSpec` they are attached into. `Cfg.driven` names the hands the action
commands; every other hand holds where it was last sent. `Cfg.camera_latency`
is the seconds a frame takes to reach a policy once the shutter has fallen,
the sensor's own delay and nothing else; a free-running lens carries the frame
in hand up to one frame period past it. The envs' data are the truth between
steps: anything moved on `rig.datas[i]` is what the next step runs from, and
an env puts its own things there between `place` and `begin`. After a step the
rig reports `unstable`, whether the simulation went unstable, and
`tracking_error`, the largest gap between a driven hand and its setpoint;
every row's joint coordinates sit in `rig.qpos_rows` and `rig.qvel_rows`.

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
five by default, and discards the rest; a policy predicts a chunk of
`CHUNK_SIZE`, sixteen, so it commits to more motion than one step executes
and resolves it finer than one step could. The setpoint walks the
interpolated path at the physics rate with the velocity of the segment it is
on, and the embodiment chases it every physics step. `action.hold` builds the
action chunk that asks to stay put.

Nothing synchronizes a lens to any of this. Each fires at its own rate on the
physics lattice, from wherever in its period an episode opens it, so it slips
against the control step and against the other lens, and its shutter falls
between rows as a capture's does. An episode draws where each lens stands from
the seed it is reproduced by. The observation carries the newest exposure
older than `camera_latency`, beside the row it was taken at; a dataset takes
every exposure from `Rig.exposed`.

### Choosing the embodiment

`Cfg.embodiment` names what holds the grippers up and which slots,
`left_gripper` and `right_gripper`, the scene holds. Another embodiment
implements `specialist_sim.embodiment.Embodiment`: it stands the hands in the
scene, binds their ids, puts a hand at a pose, and writes `ctrl` and `xfrc`
for every env each physics step. A hand's state is its base tag's pose
whatever carries it, so a policy is not told. `tests/gantry.py` is such an
embodiment, a joint chain driven through `ctrl`, and the seam tests run on it
beside the free rig.

### Choosing the cameras

Every lens is configuration, defaulting to the nominal camera:

```python
from specialist_sim.camera import Camera

Cfg(scene=..., cameras={"right_gripper": measured})
```

## Evaluation

`eval.eval_policy` runs one episode per seed on a `Policy` through any `Env`
that offers `reset` and `step` returning an observation and a report with
`success`, `done`, `unstable` and `tracking_error`, keeping every env busy. An
env counts whatever else it likes under its own names, in `report.labels`, and
an outcome carries them; evaluation rates each one without knowing what any of
them mean. `eval.summary` reports those rates beside everything they were
produced under, including a `revision` that hashes the gripper, the
configuration, the physics engine and the files under the env's `assets`.

## Demonstrations

`scripted.ScriptedPolicy` carries out a `Script`: stages that each move one
hand along a minimum-jerk curve to a waypoint the script names, work the
fingers over a window of the move, turn the wrist on the way, and hand their
speed to the next. Every waypoint is read off the env state when a stage
begins, and a stage ends when the plant has answered, not when its move has.
An env supplies the script.

## The pick-and-place env

```python
from specialist_sim.pickplace import Cfg, PickPlaceEnv
from specialist_sim.action import hold

env = PickPlaceEnv()
observation, report = env.reset([0])
observation, report = env.step(hold(observation.state[:, -1]))
```

Two hands stand at a table with a tape target and a cube, in the room the
recordings were made in: the two tag stands' midpoint is the origin, the
tabletop starts 0.40 m past the tag line, and the floor, walls and tabletop
carry the diffuse maps of the room's carpet, plaster and laminate.
`Cfg(fiducials=True)` stands the tag stands in it, their `tagStandard41h12`
patterns laid as boxes.

`report` carries the task's predicates on every env's cube (`cube`, `on_tape`,
`resting`, `held`, `dropped`, `displaced`, `picked`) and how the episode
stands (`success`, `unstable`, `tracking_error`, `reward`, `done`). The reward
is one on the step the cube has settled on the tape for five steps, having
been off it, so an undiscounted return is the success indicator.

`Task` names the hands a policy drives, its initial conditions, what the cube
must do for it to be won, and any lens the observation carries beyond the
driven hands' own. `BIMANUAL` drives both hands from the tape until the cube
is put back on it; `right_task(region)` drives the right hand from a cube
kicked into `region`; `left_task(region)` drives the left until the cube rests
in `region`. A region is a convex ring less a disc. An `InitialCondition` of
your own repeats an episode exactly:

```python
from specialist_sim.pickplace.task import Fixed

env.reset([0], initial_condition=Fixed(hands, cube))   # hands (2, 8); cube (x, y, yaw)
```

`pickplace.script.scripted_policy(env, rng)` is the env's demonstrator: one
hand kicks the cube off the tape and the other picks it up and puts it back,
over and over, its style measured off the human recordings.

## Writing episodes as a dataset

With the `dataset` extra, a demonstration is written as a state layer a
trainer reads exactly as it reads a capture:

```
uv run --extra dataset python scripts/synthesize.py --dataset ~/corpus_sim --episodes 20
```

## Replaying a recorded take

The `dataset` extra also reads a solved session and drives the env from it,
drawing each simulated view over the footage it came from with every body
outlined:

```
uv run --extra dataset python scripts/replay.py \
  --dataset ~/corpus --session canon002 --output overlay.mp4
```

## Initial-condition regions from a dataset

`dataset.region` derives the region a policy's picks begin in from the dataset
it was trained on: the convex ring around every closing of the fingers on the
cube, less the disc around the tape those closings keep clear of.
