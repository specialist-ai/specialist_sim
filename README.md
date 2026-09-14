# specialist_sim

A simulated capture rig for evaluating manipulation policies.

## Setup

You need Python 3.14 and MuJoCo 3.17 or later.

```bash
uv sync
uv run pytest
```

## The lenses

A `Lens` carries the MEI intrinsics in units of the image width (`params`), the
disc it keeps (`mask_radius`) and its seat on the gripper base. `Lens()` is the
rig's own.
