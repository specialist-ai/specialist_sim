# specialist_sim

A simulated capture rig for collecting demonstrations and evaluating
manipulation policies: grippers on their embodiment, standing in whatever scene
an env puts them in. Two envs come with it, LEGO sorting at the 0927 desk and
pick-and-place. The first-choice way to collect demonstrations is
teleoperation with a Quest. A scripted policy is the fallback.

## Setup

You need Python 3.12 and a CUDA GPU for the lenses, which Warp traces. A rig
with its lenses off runs on the CPU alone.

```bash
uv sync --all-extras
uv run pytest
```

The `teleop` extra adds the Quest page's socket. The `dataset` extra adds the
corpus layer writer and the take replay.

## Terms

The names follow LeRobot, Gymnasium and the robot-learning papers (LBM, ABC,
UMI, ACT) where they name the same thing, and the Specialist corpus where they
do not.

| Term | Meaning |
| --- | --- |
| env, `num_envs`, `env_ids` | One copy of a scene, how many a vector env steps together, and which |
| episode | One trial from a seed to success, failure or the step budget, and its recording |
| rollout | Running a policy through one episode |
| initial condition | Where an episode starts, drawn from its seed |
| success, progress | Whether the task was done, and the share of it done (LBM's task completion) |
| action chunk, action | What a policy predicts at once (`CHUNK_SIZE`), and one element of it |
| `n_action_steps` | The actions of a chunk one control step executes |
| setpoint | Where the embodiment pulls a hand at each physics step, interpolated along the actions |
| row | One tick of the lattice a dataset records a hand at |
| hand, slot | A gripper, and its name: `left_gripper`, `right_gripper` |
| embodiment | What holds the grippers up: free rigs by default |
| base tag | The frame a capture tracks a hand by: a hand's state |
| lens | The gripper camera's rear fisheye |
| opening | The left finger's angle, radians (not a width) |
| operator | The person who teleoperates |
| scripted policy | A demonstrator that reads the env's true state |
| camera latency | Seconds from a shutter to the frame reaching the policy |

## The lenses

`Camera` carries the MEI intrinsics at the width they were measured at, the
lens's seat on the gripper base, and the rate it fires at. `NOMINAL` is the
rig's own. A lens keeps the disc inscribed in its frame and is black outside
it, as a recorded image does. `render.Renderer` draws any camera on any body,
and `Renderer.render_many` draws many envs' views in one batch, each exactly
as that env alone draws it.

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
`wxyz` quaternion, and its left finger's angle in radians. The base tag sits
`gripper.TAG_OFFSET` (25 mm) behind the described base, as specialist_rlft's
replay of the 0927 takes fit it. Every array has a leading axis over the
`num_envs` parallel envs the rig holds.

| field | shape | meaning |
| --- | --- | --- |
| `observation.row` | `(num_envs, rows)` int64 | the rows this step spanned |
| `observation.state` | `(num_envs, rows, driven, 8)` | each driven hand at each row |
| `observation.image` | `(num_envs, lenses, frames, size, size, 3)` uint8 | each lens's newest frames, newest last, black outside the disc |
| `observation.body` | `(num_envs, lenses, frames, size, size)` int32 | the body each pixel struck, `-1` for none |
| `observation.image_row` | `(num_envs, lenses, frames)` int64 | the row each frame was taken at |
| `action` | `(num_envs, chunk_size, driven, 8)` | the action chunk the driven hands follow |

`Cfg.scene` is whatever the hands stand in, as a call that returns the
`MjSpec` they are attached into. `Cfg.driven` names the hands the action
commands. Every other hand holds where it was last sent. `Cfg.camera_latency`
is the seconds a frame takes to reach a policy once the shutter has fallen,
the sensor's own delay and nothing else. A free-running lens carries the frame
in hand up to one frame period past it.

The envs' data are the truth between steps: anything moved on
`rig.datas[env_idx]` is what the next step runs from, and an env puts its own
things there between `place` and `begin`. `settle` runs named envs alone, the
hands held, so a scene's loose objects come to rest before an episode opens.
Each env steps its own `MjData` with `mj_step` on a thread pool, which MuJoCo
runs without the GIL, so an env lands exactly where it would alone and keeps
its solver's warm start from one physics step to the next.

After a step the rig reports `unstable`, whether MuJoCo reset an env it could
not carry, and `tracking_error`, the largest gap between a driven hand and its
setpoint. Every row's joint coordinates sit in `rig.qpos_rows` and
`rig.qvel_rows`, its hands in `rig.state_rows` and the setpoints they chased in
`rig.setpoint_rows`. `rig.rows(env_idx)` hands one env's rows to a recorder.

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
the seed it is reproduced by. The observation carries the `Cfg.frames` newest
exposures older than `camera_latency`, beside the rows they were taken at. A
dataset takes every exposure from `Rig.exposed`. With `Cfg.record` off, the
rig traces only the exposures an observation will deliver, which changes
nothing the policy sees, and keeps none for a recorder. With `Cfg.render` off,
the rig traces nothing and needs no GPU.

### Choosing the embodiment

`Cfg.embodiment` names what holds the grippers up and which slots,
`left_gripper` and `right_gripper`, the scene holds. Another embodiment
implements `specialist_sim.embodiment.Embodiment`: it stands the hands in the
scene, binds their ids, puts a hand at a pose, and writes `ctrl` and `xfrc`
for every env each physics step. `gripper.attach` seats a gripper on its
embodiment at the base tag, and gives each hand a contact bit of its own, so
the two hands collide with each other. A hand's state is its base tag's pose
whatever carries it, so a policy is not told. `tests/gantry.py` is such an
embodiment, a joint chain driven through `ctrl`, and the seam tests run on it
beside the free rig.

The default embodiment, `Floating`, flies each gripper on a massless free
joint: an impedance of 4000 N/m and 40 N m/rad pulls it toward its setpoint,
the segment's velocity fed forward and the gripper's weight cancelled. Its pull
is capped at 25 N and 1.5 N m, a few times the gripper's 4 N weight, so a hand
stops when the fingertips meet the table instead of driving through it.

### Choosing the cameras

Every lens is configuration, defaulting to the nominal camera:

```python
from specialist_sim.camera import Camera

Cfg(scene=..., cameras={"right_gripper": measured})
```

## Physics

Every solver and contact setting is one value, `physics.Physics`, which the rig
applies to every scene and every episode records. `scripts/physics_bench.py`
measures a candidate on the LEGO env on fixed seeds. The defaults come from
MuJoCo's documentation (solver parameters, slip, integrators) and from what
MuJoCo Menagerie's grippers use. Measured on the LEGO env at 150 Hz rows:

| | specialist_rlft | Default |
| --- | --- | --- |
| Step, integrator | 1.33 ms, Euler | 0.67 ms, `implicitfast` |
| Friction | elliptic, `impratio` 10 | elliptic, `impratio` 100 |
| Pads | 7 ms, impedance 0.9 / 0.95 | 2 ms, impedance 0.95 / 0.99, priority |
| Other contacts | 4 ms, impedance 0.9 / 0.95 | 2 ms, impedance 0.95 / 0.99 |
| Object damping | 3e-4 | 3e-7 |
| Two hands clearing a 12-brick pile | 10.2x real time | 5.5x real time |
| Pads into a held brick | 6.7 mm | 0.7 mm |
| Slip of a shaken brick | 3.3 mm, 1.9° | 0.3 mm, 0.7° |
| Creep of resting bricks | 25.5 mm/s | 0.04 mm/s |
| Spin kept over 50 ms of flight | 0% | 81% |

With one setting changed from the default: `impratio` 10 raises the slip from
0.3 mm to 2.7 mm. rlft's softer pads raise the sink from 0.7 mm to 6.1 mm. An
object damping of 3e-4 keeps piles creeping at 7.9 mm/s a second after they
land, because a tipping brick turns at only 0.37 rad/s, and takes all the spin
out of a tossed brick. Euler measures like `implicitfast` here, which MuJoCo
recommends for stability under damping. The NoSlip solver stays off: two
iterations cut the slip to 0.03 mm but cost a fifth of the speed, and MuJoCo
warns about it in scenes with many contacts. A 1.33 ms step runs twice as fast
(11x real time) and sinks the pads 1.0 mm.

## Evaluation

`eval.eval_policy` runs one episode per seed on a `Policy` through any `Env`
that offers `reset` and `step` returning an observation and a report with
`success`, `progress`, `done`, `unstable` and `tracking_error`, keeping every
env busy: an env whose episode ends restarts alone on the next seed. An env
counts whatever else it likes under its own names, in `report.labels`, and an
outcome carries them. Evaluation rates each one without knowing what any of
them mean. `eval.summary` reports the success rate, the mean progress and those
rates beside everything they were produced under, including a `revision` that
hashes the gripper, the configuration, the physics engine and the files under
the env's `assets`. Given an `eval.Recording`, such as `episode.Recordings`,
`eval_policy` also keeps every episode it runs.

`eval.mcnemar` compares two evaluations of the same seeds episode by episode,
with the exact McNemar test. Pairing removes the variance between seeds, so it
detects a smaller difference than two success rates do. A rerun whose
arithmetic differs only in its last bits still flips about one episode in eight
of a LEGO evaluation, so treat a difference of under ten episodes in 150 as
noise.

## Recording demonstrations

### Teleoperation, the first choice

```bash
uv run --extra teleop python scripts/record.py --out data/episodes/NAME --desk-height 0.73
```

1. Connect the headset over USB. The run sets `adb reverse tcp:8012 tcp:8012`
   and sets it again after a reconnect.
2. Open `http://localhost:8012` on this machine. The tab is the console.
3. Open the same address in the headset's browser.
4. Press **Enter VR** (a room around you) or **Enter AR** (passthrough). The
   scene stands in front of you, the desk top `--desk-height` over the floor.

Over Wi-Fi, serve with `--cert` and `--key` and open `https://<host>:8012` in
the headset. Add `--lenses` to show both wrist lenses live.

| Action | Quest | Console |
| --- | --- | --- |
| Bind or release the gripper on that side | Squeeze the grip | — |
| Close the jaws of a bound gripper | Trigger | — |
| Paused: record. Recording: keep the episode, lay a new tray, record it. | A | R |
| Drop the episode, lay a new tray, pause | B | D |
| Recenter the scene in front of the head | X | C |
| Slide the scene along the floor | Left stick | Arrow keys |
| Raise or lower the scene, turn it about the tray | Right stick | PgUp, PgDn, Q, E |

A new tray keeps the grippers where they stand and keeps their bindings. If a
new layout puts a brick in a gripper, the run uses the next seed. The console
also sets the layout of the next trays, and its *Operator's view* looks through
the headset. `specialist_sim.teleop` is scene-agnostic: `xr` maps the headset's
space to the room and moves the scene about a focus, `binding` binds a gripper
to a controller, and `server` serves the page and its socket.

### The scripted policy, the fallback

```bash
uv run python scripts/record_scripted.py --out data/episodes/scripted_drop \
    --episodes 100 --hands right_gripper --bricks 1
```

`scripted.ScriptedPolicy` carries out a `Script`: stages that each move one
hand along a minimum-jerk curve to a waypoint the script names, work the
fingers over a window of the move, turn the wrist on the way, and hand their
speed to the next. Every waypoint is read off the env state when a stage
begins, and a stage ends when the plant has answered, not when its move has.
Each env supplies a script. The LEGO one clears 30 of 30 one-brick trays with
the right hand.

### Episodes

`data/episodes/<set>/<name>.npz` (`episode.py`) holds, per row: `timestamp`,
the hands as they stood (`state`) and the setpoints they chased (`commanded`),
both `(rows, 2, 8)` in `SLOTS` order, `qpos`, `qvel`, and a JSON `meta` with
the source, the scene and its initial condition, the physics and the final
score. No image is stored, so a change to a lens needs no new recording.

```bash
MUJOCO_GL=egl uv run python scripts/render_video.py data/episodes/NAME/EPISODE.npz --out video.mp4
```

films an episode: an overview of the desk beside each wrist lens.

## The LEGO sorting env

```python
from specialist_sim.sort_lego import Cfg, SortLegoEnv

env = SortLegoEnv(Cfg(driven=("right_gripper",), bricks=1), num_envs=32)
observation, report = env.reset(range(32))
```

The 0927 desk, ported from specialist_rlft: the TILLGANG tray, a bin of each
color with its card, and twelve 2 x 4 bricks, six of each color. A layout puts
some bricks on the tray and parks the rest in their bins: `drop` (the task: 1
to 8 bricks dropped onto the tray, none starting inside another), `single`, and
`pile`. A layout is a function of `(mode, seed, bricks)`, which each episode
records. A driven hand starts over the tray's near corner on its side, its
fingers aimed at the middle of the tray floor. A hand nobody drives waits
aside, out of every lens.

Rows run at 150 Hz, actions at 50 Hz, and a control step is 36 rows (0.24 s).
The lens fires at exactly 30 Hz with no phase, so a frame falls on every fifth
row, as the dataset build pairs frames with states. A brick is sorted in its
color's bin, missorted in the other, lost off the desk. `report.success` is
every started brick sorted, and `report.progress` the share sorted.
`Cfg.swapped` draws every brick and bin card in the other color while physics
and scoring keep each brick's own, so a policy that sorts both scenes reads the
cards, not the bins' sides.

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
is put back on it. `right_task(region)` drives the right hand from a cube
kicked into `region`. `left_task(region)` drives the left until the cube rests
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

With the `dataset` extra, a pick-and-place demonstration is written as a state
layer a trainer reads exactly as it reads a capture:

```bash
uv run --extra dataset python scripts/synthesize.py --dataset ~/corpus_sim --episodes 20
```

## Replaying a recorded take

The `dataset` extra also reads a solved session and drives the env from it,
drawing each simulated view over the footage it came from with every body
outlined:

```bash
uv run --extra dataset python scripts/replay.py \
  --dataset ~/corpus --session canon002 --output overlay.mp4
```

## Initial-condition regions from a dataset

`dataset.region` derives the region a policy's picks begin in from the dataset
it was trained on: the convex ring around every closing of the fingers on the
cube, less the disc around the tape those closings keep clear of.

## Layout

```
src/specialist_sim/
  rig.py            the hands on their embodiment in num_envs envs, and the Clock
  embodiment.py     what holds the hands up: Floating
  gripper.py        the described gripper, its base tag, its pads
  physics.py        every solver and contact setting
  camera.py         the MEI fisheye, which render.py and kernels.py trace on the GPU
  action.py         action chunks, and the Interpolator the embodiment chases
  eval.py           Env, Policy, eval_policy, summary, mcnemar
  scripted.py       ScriptedPolicy, a script's stages carried out
  episode.py        a recorded episode, its recorder, and Recordings for an evaluation
  teleop/           Quest teleoperation: xr, binding, server and page.html
  sort_lego/        the LEGO env: scene, assets, initial, task, env, script
  pickplace/        the pick-and-place env
  dataset/          corpus layers, takes and regions (dataset extra)
scripts/            record, record_scripted, render_video, physics_bench, synthesize, replay
tests/
```
