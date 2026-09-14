# specialist_sim

A simulated capture rig for evaluating manipulation policies: grippers on
their carriers, standing in whatever world a scene puts them in. What holds a
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
from specialist_sim.trajectory import hold

rig = Rig(Cfg(world=mujoco.MjSpec))     # the hands in an empty world
rig.place(0, hands)                     # hands (2, 8): position, quaternion, finger angle
observation = rig.begin([0])
observation = rig.step(hold(observation.state[:, -1]))
```

A hand is eight numbers: its gripper base's room-frame position, a rotation
as a `wxyz` quaternion, and its left finger's angle in radians. Every array
has a leading axis over the worlds the rig holds.

| field | shape | meaning |
| --- | --- | --- |
| `observation.row` | `(worlds, rows)` int64 | the rows this step spanned |
| `observation.state` | `(worlds, rows, driven, 8)` | each driven hand at each row |
| `observation.image` | `(worlds, lenses, size, size, 3)` uint8 | what each lens saw, black outside the disc |
| `observation.body` | `(worlds, lenses, size, size)` int32 | the body each pixel struck, `-1` for none |
| `observation.image_row` | `(worlds, lenses)` int64 | the row each lens's frame was taken at |
| `action` | `(worlds, poses, driven, 8)` | the trajectory the driven hands follow |

`Cfg.world` is whatever the hands stand in, as a call that returns the
`MjSpec` they are attached into. `Cfg.driven` names the hands the action
commands; every other hand holds where it was last sent. `Cfg.image_lag` is
the seconds a frame takes to reach a policy once the shutter has fallen, the
sensor's own delay and nothing else; a free-running lens carries the frame in
hand up to one frame period past it. The worlds' data are
the truth between steps: anything moved on `rig.datas[w]` is what the next
step runs from, and a scene puts its own things there between `place` and
`begin`. After a step the rig reports `broke`, whether MuJoCo could not carry
a world, and `tracking`, the largest gap between a driven hand and its
setpoint; every row's joint coordinates sit in `rig.qpos_rows` and
`rig.qvel_rows`.

### Time

Time is counted in rows, the ticks of the lattice a dataset records a hand at.
`Clock` fixes every rate as a whole number of the next finer one:

| field | default | meaning |
| --- | --- | --- |
| `physics_dt` | 1/1200 s | one physics step |
| `per_row` | 12 | physics steps in a row: 100 Hz rows |
| `per_pose` | 2 | rows in a pose of the trajectory: 50 Hz poses |
| `per_step` | 10 | rows in a control step: 10 Hz steps |

A control step tracks the poses that fall inside it, five by default, and
discards the rest; a policy names sixteen, so it commits to more motion than
one step executes and resolves it finer than one step could. The setpoint
walks the commanded path at the physics rate with the velocity of the leg it
is on, and a carrier chases it every physics step. `trajectory.hold` builds
the trajectory that asks to stay put.

Nothing synchronises a lens to any of this. Each fires at its own rate on the
physics lattice, from wherever in its period an episode opens it, so it slips
against the control step and against the other lens, and its shutter falls
between rows as a capture's does. An episode draws where each lens stands from
the seed it is reproduced by. The observation carries the newest exposure
older than `image_lag`, beside the row it was taken at; a dataset takes
every exposure from `Rig.exposed`.

### Choosing what carries the hands

`Cfg.embodiment` names what holds the grippers up and which slots,
`left_gripper` and `right_gripper`, the world holds. Another carrier
implements `specialist_sim.embodiment.Embodiment`: it stands the hands in the
world, binds their ids, puts a hand at a pose, and writes `ctrl` and `xfrc`
for every world each physics step. A hand's state is its gripper base's pose
whatever carries it, so a policy is not told. `tests/gantry.py` is such a
carrier, a joint chain driven through `ctrl`, and the seam tests run on it
beside the free rig.

### Choosing the cameras

Every lens is configuration, defaulting to the nominal camera:

```python
from specialist_sim.camera import Camera

Cfg(world=..., cameras={"right_gripper": measured})
```

## The benchmark

`bench.sweep` runs one episode per seed on an `Agent` through any scene that
offers `reset` and `step` returning an observation and a report with
`success`, `done`, `broke` and `tracking`, keeping every world busy. A scene
counts whatever else it likes under its own names, in `report.labels`, and an
outcome carries them; the bench rates each one without knowing what any of
them mean. `bench.summary` reports those rates beside everything they were
produced under, including a `revision` that hashes the gripper, the
configuration, the physics engine and the files under the scene's `assets`.

