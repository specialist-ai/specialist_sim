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
