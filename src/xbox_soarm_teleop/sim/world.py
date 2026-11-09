"""World presets for the MuJoCo simulation: lighting, sky and floor.

Presets are applied to an ``mujoco.MjSpec`` before task-scene objects are
added, so every scene (default, stack, pick-and-place) can use any world.

Presets:
    studio: Checkered floor, shadowed key light plus fill light, soft gradient
        sky. Readable default for driving.
    bright: Flat, even lighting with no shadows. Good for screen recording.
    dark:   One shadowed spot light over the workspace, dim headlight. Good for
        judging depth from shadows.
    plain:  The bare robot model with MuJoCo's default headlight (the previous
        look); task scenes add their own grey ground plane.
"""

from __future__ import annotations

import mujoco

WORLD_NAMES = ("studio", "bright", "dark", "plain")
DEFAULT_WORLD = "studio"

# Floor centred under the reachable workspace (robot base at the origin, +x forward).
_FLOOR_POS = [0.2, 0.0, 0.0]
_FLOOR_SIZE = [1.0, 1.0, 0.05]
_WORKSPACE_CENTER = [0.25, 0.0, 0.0]


def apply_world(spec: mujoco.MjSpec, world: str) -> bool:
    """Add a world preset's lights, sky and floor to a model spec.

    Args:
        spec: Model spec to modify in place (robot already loaded).
        world: One of ``WORLD_NAMES``.

    Returns:
        True if the preset added a ground plane, so task scenes should not add
        their own.

    Raises:
        ValueError: ``world`` is not a known preset.
    """
    if world not in WORLD_NAMES:
        raise ValueError(f"Unknown world '{world}'. Choose from: {', '.join(WORLD_NAMES)}")
    if world == "plain":
        return False

    spec.visual.quality.shadowsize = 4096
    if world == "studio":
        _sky(spec, top=[0.62, 0.72, 0.84], bottom=[0.95, 0.96, 0.98])
        _floor(spec, rgb1=[0.82, 0.83, 0.85], rgb2=[0.70, 0.71, 0.74], checker=True, reflect=0.03)
        _headlight(spec, ambient=0.12, diffuse=0.12)
        _directional(spec, "key_light", direction=[0.35, 0.45, -1.0], diffuse=0.8, shadow=True)
        _directional(spec, "fill_light", direction=[-0.5, -0.6, -0.6], diffuse=0.2, shadow=False)
    elif world == "bright":
        _sky(spec, top=[0.96, 0.97, 0.98], bottom=[0.96, 0.97, 0.98])
        _floor(spec, rgb1=[0.90, 0.90, 0.90], rgb2=[0.90, 0.90, 0.90], checker=False, reflect=0.0)
        _headlight(spec, ambient=0.45, diffuse=0.45)
        _directional(spec, "top_light", direction=[0.0, 0.0, -1.0], diffuse=0.35, shadow=False)
        _directional(spec, "front_light", direction=[-1.0, 0.0, -0.5], diffuse=0.25, shadow=False)
    else:  # dark
        _sky(spec, top=[0.02, 0.02, 0.05], bottom=[0.07, 0.08, 0.12])
        _floor(spec, rgb1=[0.16, 0.16, 0.18], rgb2=[0.12, 0.12, 0.14], checker=True, reflect=0.15)
        _headlight(spec, ambient=0.02, diffuse=0.05)
        spot = spec.worldbody.add_light()
        spot.name = "spot_light"
        spot.type = mujoco.mjtLightType.mjLIGHT_SPOT
        spot.pos = [_WORKSPACE_CENTER[0], _WORKSPACE_CENTER[1], 1.2]
        spot.dir = [0.0, 0.0, -1.0]
        spot.cutoff = 35.0
        spot.exponent = 8.0
        spot.diffuse = [1.0, 0.97, 0.9]
        spot.specular = [0.3, 0.3, 0.3]
        spot.castshadow = True
    return True


def _headlight(spec: mujoco.MjSpec, *, ambient: float, diffuse: float) -> None:
    spec.visual.headlight.ambient = [ambient] * 3
    spec.visual.headlight.diffuse = [diffuse] * 3
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]


def _directional(
    spec: mujoco.MjSpec, name: str, *, direction: list[float], diffuse: float, shadow: bool
) -> None:
    light = spec.worldbody.add_light()
    light.name = name
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = [0.0, 0.0, 1.5]
    light.dir = direction
    light.diffuse = [diffuse] * 3
    light.specular = [0.2, 0.2, 0.2]
    light.castshadow = shadow


def _sky(spec: mujoco.MjSpec, *, top: list[float], bottom: list[float]) -> None:
    sky = spec.add_texture()
    sky.name = "world_sky"
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = top
    sky.rgb2 = bottom
    sky.width = 512
    sky.height = 3072


def _floor(
    spec: mujoco.MjSpec,
    *,
    rgb1: list[float],
    rgb2: list[float],
    checker: bool,
    reflect: float,
) -> None:
    tex = spec.add_texture()
    tex.name = "world_floor_tex"
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER if checker else mujoco.mjtBuiltin.mjBUILTIN_FLAT
    tex.rgb1 = rgb1
    tex.rgb2 = rgb2
    tex.width = 512
    tex.height = 512

    mat = spec.add_material()
    mat.name = "world_floor_mat"
    textures = list(mat.textures)
    textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "world_floor_tex"
    mat.textures = textures
    # Uniform repeat is per meter; each checker tile holds 2x2 squares -> 5 cm squares.
    mat.texrepeat = [10.0, 10.0]
    mat.texuniform = True
    mat.reflectance = reflect

    floor = spec.worldbody.add_geom()
    floor.name = "world_floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = _FLOOR_SIZE
    floor.pos = _FLOOR_POS
    floor.material = "world_floor_mat"
    floor.friction = [1.1, 0.02, 0.001]
