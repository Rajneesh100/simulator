"""Single source of truth for global physics constants.

Scenes and robots place geometry; they never override these. If a scene seems to need
different global physics, that's a signal the parameter belongs on the scene/robot
(e.g. per-geom friction for an "icy patch"), not here.
"""

TIMESTEP = 0.002  # seconds; 500 Hz physics
GRAVITY = (0.0, 0.0, -9.81)

# MuJoCo solver settings (see MuJoCo docs: <option>).
INTEGRATOR = "implicitfast"
SOLVER = "newton"
SOLVER_ITERATIONS = 50
SOLVER_TOLERANCE = 1e-8
CONE = "elliptic"  # smoother friction cone, better for wheeled contact
JACOBIAN = "auto"

# Default contact / friction parameters, applied to ground and generic geoms that don't
# specify their own. Triplet is (sliding, torsional, rolling) per MuJoCo convention.
DEFAULT_FRICTION = (1.0, 0.005, 0.0001)
DEFAULT_CONTACT_SOLREF = (0.02, 1.0)
DEFAULT_CONTACT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)

# Wheel geoms get higher tangential friction than the generic default so driven wheels grip
# rather than slide (idea.md M1 acceptance: "wheels have proper friction").
WHEEL_FRICTION = (1.2, 0.008, 0.0001)

CONTROL_HZ = 20  # policy/action rate; physics substeps per control step = 1/(TIMESTEP*CONTROL_HZ)


def physics_steps_per_control_step() -> int:
    steps = round(1.0 / (TIMESTEP * CONTROL_HZ))
    return max(1, steps)


def apply_to_spec(spec) -> None:
    """Apply the global physics constants to a freshly-created mujoco.MjSpec."""
    spec.option.timestep = TIMESTEP
    spec.option.gravity = GRAVITY
    spec.option.integrator = _integrator_enum()
    spec.option.cone = _cone_enum()
    spec.option.tolerance = SOLVER_TOLERANCE
    spec.option.iterations = SOLVER_ITERATIONS


def _integrator_enum():
    import mujoco

    return {
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
        "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }[INTEGRATOR]


def _cone_enum():
    import mujoco

    return {
        "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
        "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
    }[CONE]
