"""Synthetic YOLO dataset generation from the MuJoCo pick-and-place scene.

Renders the simulation with randomized object poses, lighting, and camera
viewpoints, and computes pixel-perfect YOLO-format labels by projecting each
object's 3D bounds through the camera — no manual annotation, unlimited data,
domain randomization for sim-to-real transfer.

    python -m rex_tendon.training.vision.generate_synthetic_data \
        --count 500 --out datasets/cubes_synth

Then fine-tune:
    python -m rex_tendon vision train datasets/cubes_synth/dataset.yaml

Classes come from the scene-object registry (color + shape), so the dataset
automatically matches whatever scene is loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from ...perception.scene_objects import SceneObject, discover_objects

# Table area objects may occupy (matches the env's spawn bounds, with margin).
SPAWN_MIN = np.array([-0.16, -0.20])
SPAWN_MAX = np.array([0.16, 0.05])
MIN_OBJECT_GAP = 0.045  # m between object centres
PARK_POSITION = np.array([0.0, 0.0, -2.0])  # hide deselected objects

# Camera pose randomization (spherical, looking at the table centre)
CAM_DIST_RANGE = (0.45, 0.85)
CAM_ELEV_RANGE = (25.0, 65.0)  # degrees above the table plane
CAM_AZIM_RANGE = (-180.0, 180.0)
CAM_LOOKAT_JITTER = 0.04

LIGHT_JITTER = 0.4  # light position jitter (m)
DIFFUSE_RANGE = (0.5, 1.1)  # light diffuse scaling


def _geom_box_corners(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int):
    """World-space corners of a geom's bounding box (box approximation)."""
    gtype = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = np.array([size[0], size[1], size[2]])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        half = np.array([size[0], size[0], size[1]])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        half = np.array([size[0], size[0], size[1] + size[0]])
    else:
        half = np.array([size[0]] * 3)
    corners = np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    ) * half
    rot = data.geom_xmat[geom_id].reshape(3, 3)
    pos = data.geom_xpos[geom_id]
    return (corners @ rot.T) + pos


def _project(points: np.ndarray, cam_pos, cam_mat, fovy_deg, w, h):
    """Pinhole projection of world points into pixels (MuJoCo camera frame).

    The camera looks along its local -z axis. Returns (pixels, depths).
    """
    rel = (points - cam_pos) @ cam_mat  # world -> camera frame (mat columns)
    depth = -rel[:, 2]
    f = 0.5 * h / np.tan(0.5 * np.radians(fovy_deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = w / 2.0 + f * rel[:, 0] / depth
        v = h / 2.0 - f * rel[:, 1] / depth
    return np.column_stack([u, v]), depth


def _yolo_bbox(pixels: np.ndarray, depths: np.ndarray, w: int, h: int):
    """Clip projected corners to the image; return YOLO (cx, cy, bw, bh) or None."""
    if np.any(depths <= 0.01):  # behind or grazing the camera
        return None
    x0, y0 = pixels.min(axis=0)
    x1, y1 = pixels.max(axis=0)
    x0, x1 = max(x0, 0.0), min(x1, float(w))
    y0, y1 = max(y0, 0.0), min(y1, float(h))
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:  # too small / outside the frame
        return None
    return ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, bw / w, bh / h)


class SyntheticSceneRenderer:
    """Randomize the scene and render (image, labels) samples."""

    def __init__(self, xml_path: str, width: int = 640, height: int = 640):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.objects = discover_objects(self.model)
        # Unique class names: distinct objects with the same color+shape (e.g.
        # two purple cubes) must share one detector class.
        self.classes: list[str] = []
        self.class_of_object: list[int] = []
        for o in self.objects:
            name = f"{o.color}_{o.shape}"
            if name not in self.classes:
                self.classes.append(name)
            self.class_of_object.append(self.classes.index(name))
        self.fovy = float(self.model.vis.global_.fovy)
        self._nominal_light_pos = self.model.light_pos.copy()
        self._nominal_diffuse = self.model.light_diffuse.copy()

    def _set_object_pose(self, obj: SceneObject, xy, yaw: float) -> None:
        jnt = self.model.body_jntadr[obj.body_id]
        if jnt < 0:
            return
        adr = self.model.jnt_qposadr[jnt]
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], obj.half_height + 0.001]
        self.data.qpos[adr + 3 : adr + 7] = [
            np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2),
        ]
        vadr = self.model.jnt_dofadr[jnt]
        if vadr >= 0:
            self.data.qvel[vadr : vadr + 6] = 0.0

    def _park_object(self, obj: SceneObject) -> None:
        jnt = self.model.body_jntadr[obj.body_id]
        if jnt < 0:
            return
        adr = self.model.jnt_qposadr[jnt]
        self.data.qpos[adr : adr + 3] = PARK_POSITION
        vadr = self.model.jnt_dofadr[jnt]
        if vadr >= 0:
            self.data.qvel[vadr : vadr + 6] = 0.0

    def _randomize(self, rng: np.random.Generator) -> list[int]:
        """Randomize object selection/poses, arm pose, and lighting.

        Returns the indices of visible (placed) objects.
        """
        mujoco.mj_resetData(self.model, self.data)

        n_visible = int(rng.integers(2, len(self.objects) + 1))
        visible = list(rng.choice(len(self.objects), n_visible, replace=False))

        placed: list[np.ndarray] = []
        for i, obj in enumerate(self.objects):
            if i not in visible:
                self._park_object(obj)
                continue
            for _ in range(50):  # rejection-sample a collision-free spot
                xy = rng.uniform(SPAWN_MIN, SPAWN_MAX)
                if all(np.linalg.norm(xy - p) >= MIN_OBJECT_GAP for p in placed):
                    break
            placed.append(xy)
            self._set_object_pose(obj, xy, float(rng.uniform(0, 2 * np.pi)))

        # Random arm pose (the arm is part of the visual scene, never labeled).
        self.data.ctrl[:] = rng.uniform(
            self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        )

        # Lighting randomization
        self.model.light_pos[:] = self._nominal_light_pos + rng.uniform(
            -LIGHT_JITTER, LIGHT_JITTER, self._nominal_light_pos.shape
        )
        self.model.light_diffuse[:] = np.clip(
            self._nominal_diffuse * rng.uniform(*DIFFUSE_RANGE), 0.0, 1.5
        )

        # Settle physics briefly so objects rest naturally.
        for _ in range(60):
            mujoco.mj_step(self.model, self.data)
        return visible

    def _random_camera(self, rng: np.random.Generator) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = float(rng.uniform(*CAM_DIST_RANGE))
        cam.elevation = -float(rng.uniform(*CAM_ELEV_RANGE))
        cam.azimuth = float(rng.uniform(*CAM_AZIM_RANGE))
        cam.lookat[:] = [
            rng.uniform(-CAM_LOOKAT_JITTER, CAM_LOOKAT_JITTER),
            rng.uniform(-CAM_LOOKAT_JITTER, CAM_LOOKAT_JITTER) - 0.06,
            0.02,
        ]
        return cam

    def sample(self, rng: np.random.Generator):
        """One randomized (bgr_image, [(class_id, cx, cy, w, h), ...]) sample."""
        visible = self._randomize(rng)
        cam = self._random_camera(rng)
        self.renderer.update_scene(self.data, camera=cam)
        rgb = self.renderer.render()

        # Camera world pose for projection: recover from the scene's camera.
        # MjvCamera (free) -> compute pos/mat from lookat/azimuth/elevation.
        az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
        forward = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        cam_pos = np.asarray(cam.lookat) - forward * cam.distance
        zaxis = -forward  # camera looks along -z
        world_up = np.array([0.0, 0.0, 1.0])
        xaxis = np.cross(world_up, zaxis)
        xaxis /= np.linalg.norm(xaxis) + 1e-12
        yaxis = np.cross(zaxis, xaxis)
        cam_mat = np.column_stack([xaxis, yaxis, zaxis])

        labels = []
        for i in visible:
            obj = self.objects[i]
            corners = _geom_box_corners(self.model, self.data, obj.geom_id)
            pixels, depths = _project(
                corners, cam_pos, cam_mat, self.fovy, self.width, self.height
            )
            bbox = _yolo_bbox(pixels, depths, self.width, self.height)
            if bbox is not None:
                labels.append((self.class_of_object[i], *bbox))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels

    def close(self) -> None:
        self.renderer.close()


def generate_dataset(
    out_dir: Path,
    count: int,
    xml_path: str = "rex_assets/rex_simulation/pick_and_place_scene.xml",
    val_split: float = 0.15,
    width: int = 640,
    height: int = 640,
    seed: int = 0,
    min_labels: int = 1,
) -> dict:
    """Generate a YOLO dataset (images/ labels/ dataset.yaml) from the sim."""
    rng = np.random.default_rng(seed)
    gen = SyntheticSceneRenderer(xml_path, width=width, height=height)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(count * val_split))
    written = {"train": 0, "val": 0}
    idx = 0
    while idx < count:
        image, labels = gen.sample(rng)
        if len(labels) < min_labels:
            continue  # nothing visible — resample
        split = "val" if idx < n_val else "train"
        stem = f"synth_{idx:05d}"
        cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), image)
        with open(out_dir / "labels" / split / f"{stem}.txt", "w") as f:
            for cls, cx, cy, bw, bh in labels:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        written[split] += 1
        idx += 1

    yaml_path = out_dir / "dataset.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(gen.classes))
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    gen.close()
    return {"written": written, "classes": gen.classes, "yaml": str(yaml_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic YOLO dataset from the MuJoCo scene"
    )
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--out", default="datasets/cubes_synth")
    parser.add_argument(
        "--xml", default="rex_assets/rex_simulation/pick_and_place_scene.xml"
    )
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = generate_dataset(
        Path(args.out),
        count=args.count,
        xml_path=args.xml,
        val_split=args.val_split,
        width=args.size,
        height=args.size,
        seed=args.seed,
    )
    print(f"Classes: {result['classes']}")
    print(f"Written: {result['written']}")
    print(f"Dataset config: {result['yaml']}")


if __name__ == "__main__":
    main()
