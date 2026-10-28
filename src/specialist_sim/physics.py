"""Every solver and contact setting a scene runs under, applied to the compiled
model and recorded in every episode. The step length is the clock's.

``pad_*`` is every contact a gripping pad makes, and the pads take priority.
``contact_*`` is every other colliding geom. ``solref`` is ``(time constant,
damping ratio)``, ``solimp`` MuJoCo's five impedance numbers.
``scripts/physics_bench.py`` measures a setting, and the README has the
defaults' evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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
  iterations: int = 100
  tolerance: float = 1e-8
  noslip_iterations: int = 0
  impratio: float = 100.0
  """Hardens the elliptic cones' friction against the normal, MuJoCo's first
  remedy for slip."""
  pad_solref: tuple[float, float] = (0.002, 1.0)
  pad_solimp: tuple[float, ...] = (0.95, 0.99, 0.001, 0.5, 2.0)
  """The time constant is the least MuJoCo allows at a 1 ms step. The
  impedance is Menagerie's Robotiq 2F-85 pads'. A pad's steady sink scales with
  ``(1 - d) / d`` times the time constant squared."""
  contact_solref: tuple[float, float] = (0.002, 1.0)
  contact_solimp: tuple[float, ...] = (0.95, 0.99, 0.001, 0.5, 2.0)
  object_damping: float = 3e-7
  """The damping of every free joint that is not an embodiment's, N m s/rad and
  N s/m. More slows a tipping object: at 3e-4 a LEGO brick's gravity torque
  turns it at only 0.37 rad/s."""

  def apply(self, model: mujoco.MjModel) -> None:
    """Set every option, contact and damping here on ``model``, in place. An
    embodiment sets its own joints' damping after."""
    model.opt.integrator = INTEGRATORS[self.integrator]
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = self.impratio
    model.opt.iterations = self.iterations
    model.opt.tolerance = self.tolerance
    model.opt.noslip_iterations = self.noslip_iterations
    colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    pads = model.geom_condim == GRIPPING
    model.geom_solref[colliding & ~pads] = self.contact_solref
    model.geom_solimp[colliding & ~pads] = self.contact_solimp
    model.geom_solref[pads] = self.pad_solref
    model.geom_solimp[pads] = self.pad_solimp
    model.geom_priority[pads] = 1
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    dofs = np.add.outer(model.jnt_dofadr[free], np.arange(6)).ravel()
    model.dof_damping[dofs] = self.object_damping

  def as_dict(self) -> dict[str, Any]:
    return asdict(self)
