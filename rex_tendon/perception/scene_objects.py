"""Scene-object registry: discover graspable objects from a loaded MuJoCo model.

Replaces the hardcoded body-name lists scattered through the controllers
(hand_sim_controller, voice executors) with discovery from the model itself:
every body named ``obj_*`` is registered with its color (from the body name
hint, the geom material, or the geom rgba — in that order), shape, and size.

This is what lets voice commands like "pick up the red cube" resolve against
whatever scene happens to be loaded, instead of a list frozen in code.

No camera / vision dependencies — this reads the simulation model only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

# Palette anchors for nearest-color classification of geom/material rgba.
_PALETTE: dict[str, tuple[float, float, float]] = {
    "red": (0.9, 0.15, 0.1),
    "green": (0.1, 0.85, 0.2),
    "blue": (0.1, 0.25, 0.9),
    "yellow": (1.0, 1.0, 0.2),
    "purple": (0.8, 0.2, 1.0),
    "orange": (1.0, 0.5, 0.0),
    "white": (0.95, 0.95, 0.95),
    "gray": (0.5, 0.5, 0.5),
}

_SHAPE_BY_GEOM_TYPE = {
    int(mujoco.mjtGeom.mjGEOM_BOX): "cube",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "bar",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
}


@dataclass(frozen=True)
class SceneObject:
    """One graspable object discovered in the scene."""

    name: str
    body_id: int
    geom_id: int
    color: str
    shape: str
    half_height: float

    def position(self, data: mujoco.MjData) -> np.ndarray:
        """Current world position of the object's body."""
        return data.xpos[self.body_id].copy()


def _geom_rgba(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Effective rgba of a geom: material color when assigned, else geom rgba."""
    mat_id = int(model.geom_matid[geom_id])
    if mat_id >= 0:
        return np.asarray(model.mat_rgba[mat_id][:3], dtype=np.float64)
    return np.asarray(model.geom_rgba[geom_id][:3], dtype=np.float64)


def _classify_color(name: str, rgb: np.ndarray) -> str:
    for color in _PALETTE:
        if color in name.lower():
            return color
    dists = {c: float(np.linalg.norm(rgb - np.array(v))) for c, v in _PALETTE.items()}
    return min(dists, key=dists.get)


def _half_height(model: mujoco.MjModel, geom_id: int) -> float:
    gtype = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(size[2])
    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return float(size[1])
    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return float(size[1] + size[0])
    return float(size[0])  # sphere and fallback


def discover_objects(
    model: mujoco.MjModel, prefix: str = "obj_"
) -> list[SceneObject]:
    """Find all graspable objects (bodies named ``<prefix>*``) in the model."""
    objects: list[SceneObject] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or not name.startswith(prefix):
            continue
        geom_id = next(
            (g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body_id),
            -1,
        )
        if geom_id < 0:
            continue
        rgb = _geom_rgba(model, geom_id)
        objects.append(
            SceneObject(
                name=name,
                body_id=body_id,
                geom_id=geom_id,
                color=_classify_color(name, rgb),
                shape=_SHAPE_BY_GEOM_TYPE.get(int(model.geom_type[geom_id]), "object"),
                half_height=_half_height(model, geom_id),
            )
        )
    return objects


def find_by_color(
    objects: list[SceneObject], color: str, shape: Optional[str] = None
) -> Optional[SceneObject]:
    """First object matching color (and shape, when given)."""
    for obj in objects:
        if obj.color == color and (shape is None or obj.shape == shape):
            return obj
    return None


def nearest_object(
    objects: list[SceneObject], data: mujoco.MjData, point: np.ndarray
) -> tuple[Optional[SceneObject], float]:
    """Object closest to a world point, ignoring parked ones (z < -0.5)."""
    best, best_d = None, float("inf")
    for obj in objects:
        pos = obj.position(data)
        if pos[2] < -0.5 or np.linalg.norm(pos[:2]) > 1.0:
            continue  # parked outside the workspace
        d = float(np.linalg.norm(pos - point))
        if d < best_d:
            best, best_d = obj, d
    return best, best_d


def available_colors(objects: list[SceneObject]) -> set[str]:
    """Colors present in the scene — feeds voice-command validation."""
    return {obj.color for obj in objects}
