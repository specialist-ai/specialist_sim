# specialist_sim

A simulated capture rig for collecting demonstrations and evaluating
manipulation policies: grippers on their embodiment, in whatever room a task
gives them. It comes with three tasks: LEGO sorting, LEGO disassembly and
sorting, and pick-and-place. Demonstrations are recorded with a Quest.

## Setup

You need Python 3.14, MuJoCo 3.17 or later, and a CUDA GPU for the lenses.
An env without lenses runs on the CPU.

```bash
uv sync --all-extras
uv run pytest
```

The `handheld` extra adds the Quest page's socket and `sim-record`'s command
line. The `dataset` extra adds the take replay, `sim-state` and PyAV, which
films an episode or a replay.

## The lenses

A hand's camera is drawn through one of its two lenses. A `Lens` carries the
MEI intrinsics in units of the image width (`params`), the disc it keeps
(`mask_radius`) and its seat on the gripper base. `Lens()` is the rig's own, and
`Cfg(lenses={"right": measured})` swaps one in. Every lens fires at
`shutter.CAMERA_HZ` on its own clock and is traced `Cfg.image_size` pixels
across, black outside its disc. Each pixel averages `Cfg.supersample` by
`Cfg.supersample` rays; the default 2 is the robot's 720 px frame halved.

## The env

```python
from specialist_sim.env import Cfg, Env, hold
from specialist_sim.tasks.pick_place import PickPlace

env = Env(PickPlace(), Cfg(driven=("right",)), num_envs=8)
observation, info = env.reset(range(8))
observation, info = env.step(hold(observation.state[:, -1], 16))
```

A hand is eight numbers: its base tag's position in the room, a `wxyz`
quaternion, and its opening, the right finger's angle negated, as the servo
reads it. The fingers are coupled, so it differs negligibly from the dataset's
opening, `(left - right) / 2`. Every array has a leading `num_envs` axis.

| field | shape | meaning |
| --- | --- | --- |
| `observation.row` | `(num_envs, rows)` int64 | the rows this step spanned |
| `observation.state` | `(num_envs, rows, driven, 8)` | each driven hand at each row |
| `observation.image` | `(num_envs, lenses, size, size, 3)` uint8 | each lens's newest frame |
| `observation.body` | `(num_envs, lenses, size, size)` int32 | the highest-numbered body each pixel's rays struck, `-1` for none |
| `observation.capture_row` | `(num_envs, lenses)` int64 | the row each frame is filed under |
| `action` | `(num_envs, chunk_size, driven, 8)` | the action chunk the driven hands follow |

`observation.state` and `action` follow the `Cfg.driven` order of hands, and
the lens axes the `Cfg.image_hands` order; everything else is in `HANDS` order.

`Cfg` holds the `clock`, the `embodiment`, the `lenses`, the `image_size`, the
`supersample`, the `finger` servo's stiffness and damping, the `camera_latency`
(0.12 s, the sensor's own delay), the `driven` hands, the `image_hands` and
`autoreset`. A hand not driven holds where it was last sent.

A reset lays each env from its seed, or from a `Start`. A reset and every step
return the observation and an `info` dict of `(num_envs,)` arrays:

| key | meaning |
| --- | --- |
| `labels` | the task's labels by name, the latched ones held once true |
| `counts` | the task's counts by name |
| `progress` | how far the task has come, 0 to 1 |
| `success` | progress has held at 1 for `Task.dwell` seconds, latched |
| `unstable` | MuJoCo reset the env for want of a finite state, latched |
| `tracking_error` | the largest gap between a driven hand and its setpoint |
| `done` | `success`, `Verdict.failed`, or `unstable`; evaluation owns the time limit |
| `seed` | the seed of the layout actually laid |

`Cfg.autoreset` opens seed + 1 in an ended env as its next step starts, hands
where they stand. It is on by default; a reset before that step cancels it.

`env.rows(env_idx)` hands one env's rows of the last step to a recorder,
`env.failing(env_idx)` lists the contacts in one env that `Task.fails` names,
`env.trace(qpos)` draws every lens at recorded coordinates, and
`env.describe()` is the env as JSON, which `env.make(meta, **cfg)` stands again.

### Time

Time is counted in rows. A physics step is `PHYSICS_DT`, 1/1200 s, and `Clock`
fixes every other rate as a whole number of the next finer one:

| field | default | meaning |
| --- | --- | --- |
| `per_row` | 12 | physics steps in a row: 100 Hz rows |
| `per_action` | 2 | rows in one action of a chunk: 50 Hz actions |
| `per_step` | 10 | rows in a control step: 10 Hz steps |

A control step executes the first `n_action_steps` actions of a chunk, five by
default. `hold` builds a chunk that stays put, and `ROWWISE` is one row a step,
for a hand that follows a person live.

### The embodiment

`Cfg.embodiment` names the class that holds both hands up. The default,
`Floating`, flies each gripper on a massless free joint and pulls it toward its
setpoint with six impedance actuators. Another embodiment implements
`embodiment.Embodiment`; `tests/fixtures/gantry.py` is one.

### The task

A `Task` is a frozen dataclass whose fields an episode records. `spec` builds
its room, `draw` lays an episode from a seed, and `judge` turns the `Objects` of
every env into a `Verdict`: its progress, whether it failed, labels and counts.
The goal holds where progress is 1; `Task.dwell` is the seconds it must hold
before success latches. `Task.latched` names the labels held once true.
`Task.fails` names, by object, the geom name prefixes that `Objects.failed`
reports it touching. `Task.bonds` lists the bonds between objects, which a
reset activates and a load breaks, and `Objects.bonded` says which still hold.
`Task()` is the lit empty room.

### Bonds

A `Bond` is a weld that holds one object, the part, on another, the base. A
reset activates it where the episode lays the part at `Bond.pose` in the base's
frame, within the distance and angle in `ENGAGED`. The part rests on a face, a
rectangle of half sizes `Bond.face` about `Bond.anchor`, normal to the base's z
axis.

After every row, the env sums what the base applies to the part, through the
weld and through their contacts. `N` is the force pulling the part off the
base, and `M` is the moment about the anchor, both in the base's frame. The env
breaks the bond for good once

    N + max(|M_y| / face_x, |M_x| / face_y) >= pull

The second term divides the moment by the distance to the edge the part would
tip about, so prying the part up about an edge breaks the bond as a straight
pull does. A push into the face never breaks a bond, and neither does a shear
or a twist. The env does not check bonds while a reset settles the room.

Each episode draws every bond's pull from a uniform distribution between the
two values of `Bond.pull`. The draw uses a separate stream spawned from the
episode's seed, so bonds leave the layout's other draws unchanged. `env.pulls`
holds the draws, shaped `(num_envs, bonds)`.

Reading the load copies the env's `MjData` and solves its constraints again.
It costs about 0.09 ms per env per row while any of the env's bonds holds. A
row of 32 envs of the LEGO room takes 4.8 ms with three pairs each, and 2.0 ms
without bonds.

## Physics

One set of solver and contact settings applies to every room: `IMPRATIO`,
`SOLREF` and `SOLIMP`, in `env.py`. `BOND_SOLIMP` sets a bond's weld, which
stretches 0.003 mm per newton.
`tests/test_physics_settings.py` holds them to what they are for.

## Evaluation

`eval.eval_policy(env, policy, seeds, time_limit)` runs one episode per seed on
a `Policy`, each for `time_limit` seconds counted in whole control steps, at
least one, keeping every env busy. `eval.summary` reports `success_rate`,
`progress`, `unstable_rate`, a rate for each label, `time_to_success` in seconds
and `tracking_error`, beside the `time_limit`, `n_episodes`, `env.describe()`
and a `revision` that hashes the compiled model, the description and the
source. A rollout is deterministic: the same seeds give the same outcomes.

## Recording demonstrations

```bash
uv run --extra handheld sim-record sort-lego --out ~/recordings/NAME
uv run --extra handheld sim-record disassemble-sort-lego --out ~/recordings/NAME
uv run --extra handheld sim-record pick-place --out ~/recordings/NAME
```

Each task is a subcommand and its fields are its flags, `sim-record sort-lego
--mode pile`.

1. Connect the headset over USB; the run keeps `adb reverse tcp:8012 tcp:8012`
   in place. Over Wi-Fi, serve with `--cert` and `--key`.
2. Open `http://localhost:8012` in the headset's browser. The same address on
   this machine shows the scene and the status.
3. Press **Enter VR** or **Enter AR**. Each gripper rides its controller, held
   by its handle.

The left controller starts and the right one ends.

| Action | Quest |
| --- | --- |
| Close the jaws | Trigger |
| Paused: record | X |
| Recenter the scene in front of the head | Y |
| Recording: keep the episode, lay a new layout, pause | A |
| Drop any episode, lay the layout again from its start, pause | B |

An episode that ends on its own is kept, less its last `--cut` seconds if it
failed, and the next layout follows. One without progress is never kept.

A reset blocks the loop for about 60 ms while the operator keeps moving the
controllers. The hands then take about as long again to reach the controllers'
new poses. No episode keeps the rows within `--lead` seconds after a reset, 0.1 s
by default.

The fingers run on `xr.FINGER`, a servo critically damped at 25 Hz for a
demonstrator's grip.

### Episodes

`<name>.npz` (`episode.py`) holds, per row, the `timestamp`, the hands as they
stood (`state`), the setpoints they chased (`setpoint`) and `qpos`, and a JSON
`meta` that `env.make` stands the env again from. No image is
stored. `episode.joined` puts an episode's steps end to end.

A recorded episode's `meta` holds `env.describe()` and the keys below. They
describe the env at the episode's last row, even when a failed episode drops its
last `--cut` seconds.

| key | meaning |
| --- | --- |
| `score` | the task's counts |
| `seed` | the seed the layout was drawn from |
| `pulls` | every bond's pull in newtons, from `env.pulls`, active or not |
| `ended` | `kept`, `success` or `failed` |
| `failing` | the contacts the episode failed on, each `[body, geom]`, from `env.failing` |
| `unstable` | whether MuJoCo reset the env |

```bash
MUJOCO_GL=egl uv run --extra dataset sim-render ~/recordings/NAME/EPISODE.npz --out video.mp4
```

films an episode: an overview of its scene beside each hand's wrist lens.

```bash
uv run --extra dataset sim-state ~/recordings/NAME/*.npz --output-dir ~/dataset_sim
```

writes each episode as the state layer specialist's `scripts/state.py` writes
for a capture, `<output-dir>/state_s<scale>_r<f_row>/<name>.rrd`. An image is
`--image-size` pixels across, 360 by default, and `scale` is 2880 over it.
`--hands` picks `left`, `right` or `both`, `right` by default. Each written hand
takes its rank in left, right order as its slot. Each lens fires at 29.97 Hz:
`cam00` from time 0, a later slot at an offset drawn from the episode's name.
`sim-state` traces each frame at the row nearest its shutter, its `capture_row`.
Each pixel averages `--supersample` by `--supersample` rays, 8 by default, and
each image is black outside the rig's disc.

## The LEGO sorting task

```python
from specialist_sim.env import Cfg, Env
from specialist_sim.tasks.sort_lego import SortLego

env = Env(SortLego(bricks=1), Cfg(driven=("right",)), num_envs=32)
observation, info = env.reset(range(32))
```

A desk with a tray, a red and a blue AkroBin 30210, and twelve 2 x 4 bricks.
A layout, `drop` (the task) or `pile`, puts 1 to 8 bricks on the tray and parks
the rest in their bins. A brick is sorted once it rests in its color's bin, let
go of, missorted in the other, or lost below the desk. `info["counts"]` holds `started`, `sorted`,
`missorted`, `on_tray` and `lost`; success is every started brick sorted. A
missorted brick ends the episode, as does one touching the desk top, the floor
or the other color's bin.

Every LEGO task builds its bricks with `scene.brick`. A brick has the mass and
inertia of its box, studs included. It collides as two boxes: its body below
the studs, and a slab around its studs. The slab collides with everything
except another brick's body, so a brick laid on another's studs sinks onto them
as onto its hollow underside, 9.525 mm above the lower brick. On anything else,
a brick stands on its studs.

## The LEGO disassembly and sorting task

```python
from specialist_sim.env import Env
from specialist_sim.tasks.disassemble_sort_lego import DisassembleSortLego

env = Env(DisassembleSortLego(pairs=1), num_envs=8)
observation, info = env.reset(range(8))
```

The task uses the LEGO sorting room. A reset drops 1 to 3 pairs onto the tray,
each turned at random. A pair is a red and a blue brick, either color on top,
with the top brick clutched two studs along the lower one like a stair. Each
pair's clutch is a `Bond` with a 2 x 2 stud face, and each episode draws its
pull between 30 and 40 N, the bounds in `PULL`.

Lifting the top brick by its middle breaks the bond at about half its pull. The
middle sits one stud off the face's middle, so the lift also pries. Both hands
are driven by default. One grasp pulls a pair apart and keeps its brick, even at
twice the strongest pull.

A free pair buckles apart when opposed pushes over about 20 N squeeze it across
its face's edge. A free pair also comes apart when a shear and a twist of about
40 N act on it at once.

`info["counts"]` adds `pairs` and `apart` to the sorting counts, and
`info["labels"]["apart"]` is true once every pair is apart. A brick still
clutched to its pair never counts as sorted. Progress is pairs apart plus bricks
sorted, over pairs plus bricks. The episode succeeds once every brick is sorted,
and fails as a sorting episode fails.

## The pick-and-place task

```python
from specialist_sim.env import Env, Start
from specialist_sim.tasks.pick_place import PickPlace

env = Env(PickPlace())
observation, info = env.reset([0])
env.reset([0], start=Start(hands, cube))   # hands (2, 8); cube (1, 7)
```

A cube starts off a tape target on the desk and is won when it rests on the
tape, let go of, for half a second. It is lost once it falls off the desk.
`info["labels"]` carries `picked`, latched.

## Replaying a recorded take

With the `dataset` extra, `take.Take` reads a solved session and drives the
hands of an empty room from it. `sim-replay` films each simulated view over the
footage it came from and prints how closely the hands tracked the solve:

```bash
uv run --extra dataset sim-replay \
  --dataset ~/dataset_0903 --session 20260903_030121_IAQEB2601UVVAJ \
  --layer state_s12_r100 --out overlay.mp4
```
