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
