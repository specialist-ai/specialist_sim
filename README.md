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
An observation carries every row of the 100 Hz step timeline a control step
spanned, for each hand the task drives, with a leading axis over the worlds
the environment holds:

| key | shape | meaning |
| --- | --- | --- |
| `observation["step"]` | `(worlds, rows)` int64 | the rows this step spanned |
| `observation[slot]["pose"]` | `(worlds, rows, 7)` float32 | the base's position and quaternion at each row |
| `observation[slot]["opening"]` | `(worlds, rows)` float32 | the finger angle at each row |
| `observation[slot]["image"]` | `(worlds, image_size, image_size, 3)` uint8 | what the lens saw, blanked outside the disc |
| `observation[slot]["image_step"]` | `(worlds,)` int64 | the row that frame was taken at |
| `action` | `(worlds, 16, 8 × hands)` float64 | the trajectory the driven hands follow over the next 0.32 s |

`observation.flat_row` lays one world's hands out like one row of the action,
so a residual policy is a subtraction. The reward is one on the step the cube
settles onto the tape unheld, so an undiscounted return is the success
indicator.

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

## The room behind it

The scene owns what a hand can reach -- the table, the tape, the cube -- and a
photograph owns everything else. `assets/room_skybox.png` is a 360 shot of the
capture room, and a ray that strikes no geom reads it instead of coming back
black. There are no wall geoms: a flat box painted the colour of plaster is a
worse likeness of that room than a picture of it, and it moved with the scene's
lights while the picture carries the light it was taken under.

Two consequences worth knowing before trusting a rendered frame:

- **The room has no distance.** A skybox is sampled by direction alone, so every
  surface in it is infinitely far. That is only fair because the photograph was
  taken inside the workspace, 0.19 m above the table and within 0.1 m of where a
  hand rests: over the 0.2 m a camera moves, the far wall shifts by under 8
  degrees. The near surface that would shift more is the table, and the scene
  draws that itself.
- **The photograph is not lit by the scene.** It arrives already exposed, so the
  lights shade geoms and never the sky. A geom and the wall behind it therefore
  answer to different exposures, and the two only agree by having been matched
  once by eye.

`scripts/skybox.py` rebuilds the asset from an equirectangular panorama and
carries the measurement that turns it into the room frame: both anchor tags are
in the shot, and the room is defined by them, so their bearings fix the yaw to
within about three degrees.

## Starting where you choose

`reset` takes a state and a cube position, for an evaluation that repeats one
episode exactly; the state names every hand the room holds, eight numbers each:

```python
env.reset(seed=0, options={"state": start, "cube": (0.52, 0.06)})
```
