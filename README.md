# specialist_sim

A simulated capture rig for evaluating manipulation policies. Two grippers
pick and place a cube on a table, each carrying a fisheye camera that renders
through the same lens model the real captures were solved with. What holds a
gripper up is configuration: by default each flies free, pulled about by a
wrench.

```python
from specialist_sim import PickPlaceEnv
from specialist_sim.action import hold
from specialist_sim.observation import flat_row

env = PickPlaceEnv()
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(hold(flat_row(observation)))
```

The interface follows Gymnasium; the package does not depend on it.

## Observation and action

A hand is eight numbers: its gripper base's room-frame position, a rotation
as a quaternion, and a finger opening in radians. One such vector is a pose.
An observation carries every row of the row timeline — `f_row`, 100 Hz by
default — a control step spanned, for each hand the task drives, with a leading
axis over the worlds the environment holds:

| key | shape | meaning |
| --- | --- | --- |
| `observation["row"]` | `(worlds, rows)` int64 | the rows this step spanned |
| `observation[slot]["pose"]` | `(worlds, rows, 7)` float32 | the base's position and quaternion at each row |
| `observation[slot]["opening"]` | `(worlds, rows)` float32 | the finger angle at each row |
| `observation[slot]["image"]` | `(worlds, image_size, image_size, 3)` uint8 | what the lens saw, blanked outside the disc |
| `observation[slot]["image_row"]` | `(worlds,)` int64 | the row that frame was taken at |
| `action` | `(worlds, 16, 8 × hands)` float64 | the trajectory the driven hands follow over the next 0.32 s |

`observation.flat_row` lays one world's hands out like one row of the action,
so a residual policy is a subtraction. The reward is one on the step the cube
settles onto the tape unheld, so an undiscounted return is the success
indicator.

A lens fires at `camera_hz`, 30 Hz by default, on a clock of its own: a control
step spans a whole number of its frames and its last one lands on the boundary.
An observation is handed the newest frame the lens had taken `camera_lag`
seconds back, and `image_row` says which row that frame sits on. A step draws that one
frame and no other, so the grid decides how old a frame is without deciding how
much is rendered.

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

The environment resamples the trajectory onto the physics step, so the
setpoint walks the commanded path at 2 kHz rather than jumping once per
control step, with a velocity feedforward from the leg it is on. A controller
chases that ramp every physics step; the default is a stiff impedance pulling
each free-flying gripper with a wrench.

The action and observation spaces are unbounded. A setpoint is tracked
wherever it names, and a pose and an opening report where the plant is.

The three rates are locked together: a control step must span a whole number of
poses, and a pose a whole number of physics steps. At 2 kHz physics and a 50 Hz
trajectory, the control rate has to divide 50. The horizon is `HORIZON / 50` s,
so it moves in 20 ms steps: raising `HORIZON` alone changes nothing else.

`env.observation_space` and `env.action_space` describe these as Gymnasium
objects, from the environment's own configuration. They need the `gym` extra.

`env.render()` returns a view of the room from a fixed camera when the
environment is built with `render_mode="rgb_array"`. Alongside it,
`env.views[slot].body` names the body each pixel struck.

## Choosing what carries the hands

`PickPlaceCfg.embodiment` names what holds the grippers up and which of the
two slots, `left_gripper` and `right_gripper`, the room holds. A task drives
some of those hands; a `reset` state names all of them. `PickPlaceCfg.controller`
names what drives them along the ramp, defaulting to what the embodiment prefers.

```python
from specialist_sim.embodiment import FloatingCfg
from specialist_sim.task import right_task

one = FloatingCfg(slots=("right_gripper",))
env = PickPlaceEnv(PickPlaceCfg(embodiment=one, task=right_task(region)))
```

Another carrier implements `specialist_sim.embodiment.Embodiment`, and whatever
drives it `specialist_sim.control.Controller`; the protocols say what each must
do. A hand's state is its gripper base's pose whatever carries it, so a policy
is not told. `tests/gantry.py` is such a carrier, a joint chain read by forward
kinematics and driven through `ctrl`, and the seam tests run on it beside the
free rig.

## Choosing the cameras

Every lens is configuration, defaulting to a nominal camera:

```python
from specialist_sim import PickPlaceCfg, PickPlaceEnv
from specialist_sim.calibration import CameraCfg

env = PickPlaceEnv(PickPlaceCfg(cameras={"left_gripper": CameraCfg(), "right_gripper": measured}))
```

`CameraCfg` carries the MEI intrinsics at the width they were measured at and
the lens's seat on the gripper base, so a caller holding a real calibration
renders through it. The cameras are named for the slots the embodiment holds;
left unset, every hand carries the nominal camera.

## The room around it

The camera sees world-anchored primitive geometry for the blue floor, the two
walls bounding the work area, the tabletop and its center tape. Their diffuse
maps add the carpet, painted-plaster and laminate surface detail visible in the
capture room. Moving a wrist camera therefore produces parallax against every
nearby surface. Rays that leave the measured patch read one flat background
color, which is all a scene with no ceiling and no far wall owes them.

The room frame is the measured capture frame: the midpoint of tag IDs 0 and 1
is the origin, `+X` points toward the table, `+Y` toward the left wall, and `+Z`
points up. AprilTags are excluded by default, since not every evaluation wants
fiducials in its observations. Opt into both measured stands with:

```python
from specialist_sim.scene import SceneCfg

env = PickPlaceEnv(PickPlaceCfg(scene=SceneCfg(apriltags=True)))
```

The tag centers are at `z=0`, one meter above the floor. Their exact
`tagStandard41h12` patterns are assembled from black box primitives so the
custom tracer and MuJoCo's renderer draw the same fiducials.

## Starting where you choose

`reset` takes a state and a cube position, for an evaluation that repeats one
episode exactly; the state names every hand the room holds, eight numbers each:

```python
env.reset(seed=0, options={"state": start, "cube": (0.52, 0.06)})
```
