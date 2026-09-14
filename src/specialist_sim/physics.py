"""Every solver and contact setting a scene runs under, applied to the compiled
model. The step length is the clock's.

Every colliding geom answers a contact the same way: ``solref`` is ``(time
constant, damping ratio)``, ``solimp`` MuJoCo's five impedance numbers. A
gripping pad takes priority, so a pad against anything is the pad's contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from specialist_sim.gripper import GRIPPING

__all__ = ["INTEGRATORS", "Physics"]

INTEGRATORS = {
  "euler": mujoco.mjtIntegrator.mjINT_EULER,
  "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


@dataclass(frozen=True)
class Physics:
  integrator: str = "implicitfast"
  """MuJoCo's recommendation for stability: implicit in the damping."""
  impratio: float = 100.0
  """Hardens the elliptic cones' friction against the normal, MuJoCo's first
  remedy for slip."""
  solref: tuple[float, float] = (0.002, 1.0)
  solimp: tuple[float, ...] = (0.95, 0.99, 0.001, 0.5, 2.0)
  """The time constant sits just above MuJoCo's floor of twice the step,
  1.67 ms at 1200 Hz. The impedance is Menagerie's Robotiq 2F-85 pads'. A pad's
  steady sink scales with ``(1 - d) / d`` times the time constant squared."""
  object_damping: float = 3e-7
  """The damping of every free joint that is not an embodiment's, N m s/rad and
  N s/m. More takes an object's motion out of it: at 3e-4 a tossed LEGO brick
  keeps none of its spin through 50 ms of flight."""

  def apply(self, model: mujoco.MjModel) -> None:
    """Set every option, contact and damping here on ``model``, in place. An
    embodiment sets its own joints' damping after."""
    model.opt.integrator = INTEGRATORS[self.integrator]
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = self.impratio
    colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    model.geom_solref[colliding] = self.solref
    model.geom_solimp[colliding] = self.solimp
    model.geom_priority[model.geom_condim == GRIPPING] = 1
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    dofs = np.add.outer(model.jnt_dofadr[free], np.arange(6)).ravel()
    model.dof_damping[dofs] = self.object_damping
