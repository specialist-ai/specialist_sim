# specialist_sim

A simulated capture rig for collecting demonstrations and evaluating
manipulation policies: grippers on their embodiment, following the actions a
policy predicts.

## Setup

You need Python 3.12.

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

## Layout

```
src/specialist_sim/
  gripper.py        the described gripper, its base, its pads
  embodiment.py     what holds the hands up: Floating
  action.py         action chunks, and the Interpolator the embodiment chases
tests/
```
