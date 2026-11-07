# specialist_sim

A simulated capture rig for collecting demonstrations and evaluating
manipulation policies: grippers on their embodiment, in whatever room a task
gives them. Two tasks come with it, LEGO sorting and pick-and-place, and
demonstrations are recorded with a Quest.

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
(0.12 s, the sensor's own delay), the `driven` hands and the `image_hands`. A
hand not driven holds where it was last sent.

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
| `done` | `success`, a label in `Task.fails`, or `unstable`; evaluation owns the time limit |
| `seed` | the seed of the layout actually laid |

`env.rows(env_idx)` hands one env's rows of the last step to a recorder,
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
its room, `draw` lays an episode from a seed, and `judge` turns the `Objects`
of every env into a `Verdict`, its progress, labels and counts. The goal holds
where progress is 1; `Task.dwell` is the seconds it must hold before success
latches, and `Task.latched` and `Task.fails` say which labels hold once true
and which end an episode. `Task()` is the lit empty
room.

## Physics

One set of solver and contact settings applies to every room: `IMPRATIO`,
`SOLREF` and `SOLIMP`, in `env.py`.
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
| Recording: drop the episode, record a new layout | B |

The fingers run on `xr.FINGER`, a servo critically damped at 25 Hz for a
demonstrator's grip.

### Episodes

`<name>.npz` (`episode.py`) holds, per row, the `timestamp`, the hands as they
stood (`state`), the setpoints they chased (`setpoint`) and `qpos`, and a JSON
`meta` that `env.make` stands the env again from. No image is
stored. `episode.joined` puts an episode's steps end to end.

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
the rest in their bins. A brick is sorted in its color's bin, missorted in the
other, or lost below the desk. `info["counts"]` holds `started`, `sorted`,
`missorted`, `on_tray` and `lost`; success is every started brick sorted.

## The pick-and-place task

```python
from specialist_sim.env import Env, Start
from specialist_sim.tasks.pick_place import PickPlace

env = Env(PickPlace())
observation, info = env.reset([0])
env.reset([0], start=Start(hands, cube))   # hands (2, 8); cube (1, 7)
```

A cube starts off a tape target on the desk and is won when it rests on the
tape, let go of, for half a second. `info["labels"]` carries `picked`, latched,
and `dropped`, which ends the episode.

## Replaying a recorded take

With the `dataset` extra, `take.Take` reads a solved session and drives the
hands of an empty room from it. `sim-replay` films each simulated view over the
footage it came from and prints how closely the hands tracked the solve:

```bash
uv run --extra dataset sim-replay \
  --dataset ~/dataset_0903 --session 20260903_030121_IAQEB2601UVVAJ \
  --layer state_s12_r100 --out overlay.mp4
```
