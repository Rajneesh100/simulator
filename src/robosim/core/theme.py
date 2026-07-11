"""Isaac-Sim-style visual theme for the world model — presentation only, zero physics.

Lives in core/ (not ui/) purely to keep the import graph clean: core/world.py applies it
at build time (skybox/grid are model assets, fixed at MjSpec-compile). Nothing here may
touch option/solver/friction — physics stays centralized in core/physics.py.
"""

from __future__ import annotations

import mujoco

# Viewport palette (Isaac-ish: near-black blue-gray viewport, subtle grid)
SKY_TOP = (0.09, 0.10, 0.12)
SKY_HORIZON = (0.02, 0.02, 0.03)
GRID_CHECKER_A = (0.10, 0.11, 0.13)
GRID_CHECKER_B = (0.13, 0.14, 0.16)
GRID_LINE = (0.30, 0.32, 0.36)
HAZE_RGBA = (0.04, 0.045, 0.06, 1.0)

GRID_MATERIAL = "__theme_grid"
GRID_CELL_METERS = 0.5  # checker square size; texrepeat=1/m with texuniform gives 2 squares/m

# Default camera framing (also used by the interactive viewers on launch)
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -22.0
CAMERA_DISTANCE = 2.8

OFFSCREEN_WIDTH = 1280
OFFSCREEN_HEIGHT = 720


def apply_visual_theme(spec) -> None:
    """Add dark skybox + grid material and set global visual quality on a fresh MjSpec."""
    spec.add_texture(
        name="__theme_skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=list(SKY_TOP),
        rgb2=list(SKY_HORIZON),
        width=256,
        height=1536,
    )
    spec.add_texture(
        name="__theme_grid_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=list(GRID_CHECKER_A),
        rgb2=list(GRID_CHECKER_B),
        markrgb=list(GRID_LINE),
        width=300,
        height=300,
    )
    mat = spec.add_material(name=GRID_MATERIAL, texuniform=True, reflectance=0.15)
    mat.texrepeat = [1.0 / GRID_CELL_METERS / 2.0, 1.0 / GRID_CELL_METERS / 2.0]
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "__theme_grid_tex"

    v = spec.visual
    v.headlight.ambient = [0.25, 0.25, 0.28]
    v.headlight.diffuse = [0.35, 0.35, 0.35]
    v.headlight.specular = [0.05, 0.05, 0.05]
    v.quality.shadowsize = 4096
    v.rgba.haze = list(HAZE_RGBA)
    v.map.haze = 0.3
    v.global_.azimuth = CAMERA_AZIMUTH
    v.global_.elevation = CAMERA_ELEVATION
    v.global_.offwidth = OFFSCREEN_WIDTH
    v.global_.offheight = OFFSCREEN_HEIGHT


def assign_ground_material(spec, ground_geom_names: list[str]) -> None:
    """Give the scene's ground plane(s) the grid material. Only plane geoms are themed —
    heightfield terrain keeps whatever look its scene chose."""
    for geom in spec.geoms:
        if geom.name in ground_geom_names and geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.material = GRID_MATERIAL
