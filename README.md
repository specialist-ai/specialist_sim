# specialist_sim

A simulated capture rig for collecting demonstrations and evaluating
manipulation policies: grippers on their embodiment, each watching through the
fisheye its real counterpart has.

## Setup

You need Python 3.12 and a CUDA GPU for the lenses, which Warp traces. A rig
with its lenses off runs on the CPU alone.

```bash
uv sync
uv run pytest
```

## The hands

A gripper's model comes from `specialist_description`, so the simulated hand
and the real one are the same object. `gripper.py` seats it by its base and
names its pads.

`embodiment.Floating` holds each gripper up on a massless free joint, pulling
it toward its setpoint with a stiff impedance and cancelling its weight.

A policy commands a trajectory, not a pose. `action.py` turns the action chunk
it predicts into the setpoint the embodiment chases at every physics step,
bounded by a top speed and turn rate.

## The lenses

`Camera` carries the MEI intrinsics at the width they were measured at, the
lens's seat on the gripper base. `NOMINAL` is the rig's own. A lens keeps the disc inscribed in its frame and is black outside
it, as a recorded image does. `render.Renderer` draws any camera on any body.

## Layout

```
src/specialist_sim/
  gripper.py        the described gripper, its base, its pads
  embodiment.py     what holds the hands up: Floating
  action.py         action chunks, and the Interpolator the embodiment chases
  camera.py         the MEI fisheye, which render.py and kernels.py trace on the GPU
tests/
```
