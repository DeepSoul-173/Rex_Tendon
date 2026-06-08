"""MuJoCo-based pick-and-place RL environment for tendon-driven continuum robots."""

import os
from collections import deque
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame

from ...common.constants import MOTOR_NAMES
from ...configs.pick_place_config import PickPlaceEnvConfig
from ...control.geometry import convert_2d_cursor_to_target_lengths

logger = logging.getLogger(__name__)

TABLE_TOP_GEOM_NAME = "table_top"
TIP_CONTACT_GEOM_NAME = "tip_contact"
DEFAULT_TABLE_SURFACE_Z = 0.0
OBJECT_SPAWN_CLEARANCE = 0.002
LIFT_SUCCESS_HEIGHT = 0.02


class PickPlacePhase:
    """Curriculum learning phases."""

    REACH = 0
    REACH_GRASP = 1
    FULL = 2


class TentaclePickPlaceEnv(gym.Env):
    """Gymnasium environment for pick-and-place with a tendon-driven continuum robot.

    The robot must:
    1. Reach toward a target object
    2. Grasp it (via weld constraint when tip is close enough)
    3. Transport it to the place zone
    4. Release it

    Observation: stacked frames of [tip_pos(3), obj_pos(3), place_pos(3),
                                     grasp_state(1), actuator_lengths(3), prev_action(2)]
    Action: 2D cursor [-1, 1]^2 -> converted to 3 tendon lengths
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: Optional[PickPlaceEnvConfig] = None,
        render_mode: Optional[str] = None,
        global_step_counter: Any = None,
        task_config=None,
        object_curriculum_state: Any = None,
    ):
        super().__init__()

        self.config = config or PickPlaceEnvConfig()
        self.render_mode = render_mode
        self.task_config = task_config
        # Shared (cross-process) counter holding the current curriculum object
        # count; ramped by MultiObjectCurriculumCallback. None => static count.
        self._object_curriculum_state = object_curriculum_state

        # Global step counter for curriculum learning (shared across envs)
        # Use a mutable list so it can be shared: [current_step]
        self._global_step_counter = global_step_counter or [0]

        # Load MuJoCo model
        xml_file = self.config.xml_file
        if not os.path.exists(xml_file):
            script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            abs_xml_file = os.path.join(script_dir, xml_file)
            if not os.path.exists(abs_xml_file):
                # Try from cwd
                abs_xml_file = os.path.join(os.getcwd(), xml_file)
                if not os.path.exists(abs_xml_file):
                    raise FileNotFoundError(
                        f"Cannot find MuJoCo XML file at {xml_file} or {abs_xml_file}"
                    )
            xml_file = abs_xml_file

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        # Time parameters
        self.simulation_length_seconds = self.config.simulation_length_seconds
        self.time_between_steps_seconds = self.config.time_between_steps_seconds
        self.timestep = self.model.opt.timestep
        self.frame_skip = max(1, round(self.time_between_steps_seconds / self.timestep))
        self.time_per_step = self.frame_skip * self.timestep
        self._max_episode_steps = int(
            self.simulation_length_seconds / self.time_per_step
        )

        # Config shortcuts
        self.initial_actuator_config = self.config.initial_actuator_position
        self.tip_site_name = self.config.tip_site_name
        self.num_frames = self.config.num_frames
        self.obs_buffer = deque(maxlen=self.num_frames)
        self.include_actuator_lengths_in_obs = (
            self.config.include_actuator_lengths_in_obs
        )
        self.include_relative_observations = self.config.include_relative_observations
        self.include_object_velocity_in_obs = self.config.include_object_velocity_in_obs
        self.max_2d_action_magnitude = self.config.max_2d_action_magnitude
        self.action_smoothing_alpha = np.clip(
            self.config.action_smoothing_alpha, 0.0, 1.0
        )
        self.max_action_delta_per_step = self.config.max_action_delta_per_step

        # Domain randomization
        self.randomize_dynamics = self.config.randomize_dynamics
        self.randomization_factors = self.config.randomization_factors
        self.add_observation_noise = self.config.add_observation_noise
        self.observation_noise_scale = self.config.observation_noise_scale

        # Store original dynamics for randomization
        self._original_dynamics = None
        if self.randomize_dynamics:
            self._store_original_dynamics()

        # Reward parameters
        self.reach_reward_scale = self.config.reach_reward_scale
        self.grasp_bonus = self.config.grasp_bonus
        self.transport_reward_scale = self.config.transport_reward_scale
        self.place_bonus = self.config.place_bonus
        self.drop_penalty = self.config.drop_penalty
        self.time_penalty = self.config.time_penalty
        self.action_change_penalty_scale = self.config.action_change_penalty_scale
        self.action_jerk_penalty_scale = self.config.action_jerk_penalty_scale
        self.object_progress_reward_scale = self.config.object_progress_reward_scale
        self.stable_contact_bonus_scale = self.config.stable_contact_bonus_scale
        self.grasp_proximity_bonus_scale = self.config.grasp_proximity_bonus_scale
        self.assisted_grasp_enabled = self.config.assisted_grasp_enabled
        self.assisted_grasp_follow_alpha = np.clip(
            self.config.assisted_grasp_follow_alpha, 0.0, 1.0
        )

        # Grasping parameters (hybrid trigger: real contact OR close proximity)
        self.grasp_distance_threshold = self.config.grasp_distance_threshold
        self.grasp_contact_force_threshold = self.config.grasp_contact_force_threshold
        self.grasp_consecutive_steps = self.config.grasp_consecutive_steps
        self.grasp_release_grace_steps = self.config.grasp_release_grace_steps
        self.grasp_carry_offset = self.config.grasp_carry_offset
        self.place_distance_threshold = self.config.place_distance_threshold

        # Curriculum
        self.curriculum_enabled = self.config.curriculum_enabled
        self.reach_only_steps = self.config.reach_only_steps
        self.reach_grasp_steps = self.config.reach_grasp_steps

        # Object spawn bounds
        self.object_spawn_min = np.array(self.config.object_spawn_bounds_min)
        self.object_spawn_max = np.array(self.config.object_spawn_bounds_max)
        self.num_spawned_objects = int(self.config.num_spawned_objects)
        self.object_curriculum_enabled = self.config.object_curriculum_enabled
        self.min_spawned_objects = int(self.config.min_spawned_objects)
        self.max_spawned_objects = int(self.config.max_spawned_objects)
        self.occlusion_radius = float(self.config.occlusion_radius)
        # Per-episode multi-object metrics (read by the curriculum callback).
        self._episode_num_objects = self.num_spawned_objects
        self._episode_collision = False
        self._episode_occluded = False

        # MuJoCo IDs
        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.tip_site_name
        )
        if self.tip_site_id == -1:
            raise ValueError(f"Site '{self.tip_site_name}' not found in model.")

        self.place_zone_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.config.place_zone_site_name
        )

        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )

        # Object body/site/geom/constraint IDs
        self.object_body_ids = []
        self.object_site_ids = []
        self.object_geom_ids = []
        self.grasp_constraint_ids = []
        self.tip_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, TIP_CONTACT_GEOM_NAME
        )
        if self.tip_geom_id == -1:
            logger.warning(
                "Tip contact geom '%s' not found; grasp detection cannot use contact force.",
                TIP_CONTACT_GEOM_NAME,
            )

        for i, name in enumerate(self.config.object_names):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                logger.warning(f"Body '{name}' not found in model — skipping.")
                continue
            self.object_body_ids.append(bid)

            # Matching site
            site_name = (
                self.config.object_site_names[i]
                if i < len(self.config.object_site_names)
                else None
            )
            if site_name:
                sid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
                )
                self.object_site_ids.append(sid)
            else:
                self.object_site_ids.append(-1)

            # Matching geom
            geom_name = (
                self.config.object_geom_names[i]
                if i < len(self.config.object_geom_names)
                else None
            )
            if geom_name:
                gid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                )
                if gid == -1:
                    gid = self._first_geom_id_for_body(bid)
                    logger.warning(
                        "Geom '%s' not found for body '%s'; using first body geom id %s.",
                        geom_name,
                        name,
                        gid,
                    )
                self.object_geom_ids.append(gid)
            else:
                self.object_geom_ids.append(self._first_geom_id_for_body(bid))

        if not self.object_body_ids:
            raise ValueError(
                "No valid object bodies found in model. "
                f"Checked: {self.config.object_names}"
            )

        # Equality constraints — one per found body
        self.grasp_constraint_ids = [-1] * len(self.object_body_ids)
        self.table_surface_z = self._infer_table_surface_z()
        self.object_rest_z = np.array(
            [
                self.table_surface_z
                + self._get_geom_vertical_extent(gid)
                + OBJECT_SPAWN_CLEARANCE
                for gid in self.object_geom_ids
            ],
            dtype=np.float32,
        )

        # State tracking
        self._elapsed_steps = 0
        self.current_position = None
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_previous_action = np.zeros(2, dtype=np.float32)
        self.previous_object_position = np.zeros(3, dtype=np.float32)
        self.calibrated_tendon_lengths = {name: 0.0 for name in MOTOR_NAMES}

        # Grasping state
        self.is_grasped = False
        self.grasp_proximity_count = 0
        self.grasp_contact_count = 0
        self.no_contact_grace_steps = 0
        self._last_tip_object_contact_force = 0.0
        self.active_object_idx = 0  # Which object to pick (selected at reset)
        self.grasp_triggered = False
        self.place_success = False
        self.grasp_local_offset = np.zeros(3, dtype=np.float32)
        self.grasp_local_rot = np.eye(3, dtype=np.float32)

        # Renderer
        self.image_size = (84, 84)
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(
                self.model, height=self.image_size[0], width=self.image_size[1]
            )
        else:
            self.renderer = None
        self.viewer = None

        # Action space: 2D cursor
        self.action_space = spaces.Box(
            low=-self.max_2d_action_magnitude,
            high=self.max_2d_action_magnitude,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_dim = 2

        # Actuator ranges
        # Only the 3 tendon actuators are RL-controlled. A 4th actuator (base
        # yaw) may exist in the model for teleoperation; it is held at 0 here so
        # the RL task stays a clean 2-DOF bend problem with a 3-tendon obs.
        actuator_ctrlrange = self.model.actuator_ctrlrange
        self.num_tendon_actuators = 3
        self.actuator_low = actuator_ctrlrange[: self.num_tendon_actuators, 0]
        self.actuator_high = actuator_ctrlrange[: self.num_tendon_actuators, 1]

        # Observation space
        # tip_pos(3) + obj_pos(3) + place_pos(3) + grasp_state(1) + prev_action(2) = 12
        single_frame_dim = 12
        if self.include_actuator_lengths_in_obs:
            single_frame_dim += 3  # actuator lengths
        if self.include_relative_observations:
            single_frame_dim += 6  # tip-object and object-place vectors
        if self.include_object_velocity_in_obs:
            single_frame_dim += 3
        self.single_frame_dim = single_frame_dim

        stacked_obs_shape = (self.num_frames * single_frame_dim,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=stacked_obs_shape, dtype=np.float32
        )

    def _first_geom_id_for_body(self, body_id: int) -> int:
        """Return the first collision geom attached to a body, or -1 if absent."""
        geom_adr = int(self.model.body_geomadr[body_id])
        geom_num = int(self.model.body_geomnum[body_id])
        return geom_adr if geom_adr >= 0 and geom_num > 0 else -1

    def _infer_table_surface_z(self) -> float:
        """Infer the table top surface in world coordinates."""
        table_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_TOP_GEOM_NAME
        )
        if table_geom_id < 0:
            return DEFAULT_TABLE_SURFACE_Z

        mujoco.mj_forward(self.model, self.data)
        center_z = float(self.data.geom_xpos[table_geom_id][2])
        half_height = float(self.model.geom_size[table_geom_id][2])
        return center_z + half_height

    def _get_geom_vertical_extent(self, geom_id: int) -> float:
        """Return conservative half-height of a geom in world Z."""
        if geom_id < 0:
            return 0.01

        geom_type = int(self.model.geom_type[geom_id])
        size = self.model.geom_size[geom_id]
        xmat = self.data.geom_xmat[geom_id].reshape(3, 3)

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return float(np.sum(np.abs(xmat[2, :]) * size[:3]))
        if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            axis_z = abs(float(xmat[2, 2]))
            radius = float(size[0])
            half_length = float(size[1])
            return axis_z * half_length + np.sqrt(max(0.0, 1.0 - axis_z**2)) * radius
        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis_z = abs(float(xmat[2, 2]))
            return float(size[0]) + axis_z * float(size[1])
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(size[0])

        return float(max(size[0], 0.01))

    def _object_rest_position(self, obj_idx: int, xy: np.ndarray) -> np.ndarray:
        """Build a stable object center position from an XY sample."""
        return np.array([xy[0], xy[1], self.object_rest_z[obj_idx]], dtype=np.float32)

    def _sample_nonoverlapping_object_xy(
        self, placed_xy: List[np.ndarray], min_separation: float = 0.045
    ) -> np.ndarray:
        """Sample a tabletop XY position that does not start inside another object."""
        for _ in range(100):
            xy = self.np_random.uniform(
                low=self.object_spawn_min[:2],
                high=self.object_spawn_max[:2],
            ).astype(np.float32)
            outside_robot_base = np.linalg.norm(xy) >= 0.075
            separated = all(
                np.linalg.norm(xy - prev_xy) >= min_separation
                for prev_xy in placed_xy
            )
            if outside_robot_base and separated:
                return xy

        span = self.object_spawn_max[:2] - self.object_spawn_min[:2]
        grid_cols = max(1, int(np.ceil(np.sqrt(len(placed_xy) + 1))))
        grid_row = len(placed_xy) // grid_cols
        grid_col = len(placed_xy) % grid_cols
        cell = np.array(
            [
                (grid_col + 0.5) / grid_cols,
                (grid_row + 0.5) / grid_cols,
            ],
            dtype=np.float32,
        )
        xy = (self.object_spawn_min[:2] + span * cell).astype(np.float32)
        if np.linalg.norm(xy) < 0.075:
            xy[0] = np.sign(xy[0] or 1.0) * 0.09
        return xy

    def _get_curriculum_phase(self) -> int:
        """Determine current curriculum phase based on global step count."""
        if not self.curriculum_enabled:
            return PickPlacePhase.FULL

        # Handle both list and multiprocessing.Value types
        if hasattr(self._global_step_counter, "value"):
            total_steps = self._global_step_counter.value
        else:
            total_steps = self._global_step_counter[0]

        if total_steps < self.reach_only_steps:
            return PickPlacePhase.REACH
        elif total_steps < self.reach_grasp_steps:
            return PickPlacePhase.REACH_GRASP
        else:
            return PickPlacePhase.FULL

    def _current_num_objects(self) -> int:
        """Resolve how many objects to spawn this episode (curriculum-aware)."""
        if self.object_curriculum_enabled and self._object_curriculum_state is not None:
            state = self._object_curriculum_state
            if hasattr(state, "value"):
                n = int(state.value)
            elif isinstance(state, (list, tuple)):
                n = int(state[0])
            else:
                n = int(state)
            n = int(np.clip(n, self.min_spawned_objects, self.max_spawned_objects))
        else:
            n = self.num_spawned_objects
        return max(1, min(n, len(self.object_body_ids)))

    def _compute_occlusion(self) -> bool:
        """Geometric target-clutter proxy: any other spawned object within
        occlusion_radius (XY) of the active object."""
        active_xy = self._get_object_position(self.active_object_idx)[:2]
        for i in range(self._episode_num_objects):
            if i == self.active_object_idx:
                continue
            other_xy = self._get_object_position(i)[:2]
            if float(np.linalg.norm(active_xy - other_xy)) < self.occlusion_radius:
                return True
        return False

    def _compute_collision(self) -> bool:
        """True if the tip contacts a non-active spawned object, or two spawned
        objects contact each other (clutter-induced disturbance)."""
        if self.tip_geom_id < 0:
            return False
        spawned = {
            self.object_geom_ids[i]
            for i in range(self._episode_num_objects)
            if self.object_geom_ids[i] >= 0
        }
        active_geom = (
            self.object_geom_ids[self.active_object_idx]
            if self.active_object_idx < len(self.object_geom_ids)
            else -1
        )
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            g1, g2 = int(con.geom1), int(con.geom2)
            if self.tip_geom_id in (g1, g2):
                other = g2 if g1 == self.tip_geom_id else g1
                if other in spawned and other != active_geom:
                    return True
            elif g1 in spawned and g2 in spawned:
                return True
        return False

    def _get_tip_position(self) -> np.ndarray:
        """Get tentacle tip position."""
        return self.data.site_xpos[self.tip_site_id].copy()

    def _get_object_position(self, obj_idx: int) -> np.ndarray:
        """Get object body position."""
        body_id = self.object_body_ids[obj_idx]
        return self.data.xpos[body_id].copy()

    def _get_place_zone_position(self) -> np.ndarray:
        """Get place zone position."""
        return self.data.site_xpos[self.place_zone_site_id].copy()

    def _set_object_position(self, obj_idx: int, position: np.ndarray) -> None:
        """Set object freejoint position while preserving identity orientation."""
        body_id = self.object_body_ids[obj_idx]
        jnt_adr = self.model.body_jntadr[body_id]
        if jnt_adr < 0:
            return

        position = np.asarray(position, dtype=np.float32).copy()
        min_center_z = float(self.object_rest_z[obj_idx])
        if position[2] <= self.table_surface_z + OBJECT_SPAWN_CLEARANCE:
            position[2] = min_center_z
        else:
            position[2] = max(float(position[2]), min_center_z)

        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        qvel_adr = self.model.jnt_dofadr[jnt_adr]
        self.data.qpos[qpos_adr : qpos_adr + 3] = position
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0

    def _set_place_zone_position(self, position: np.ndarray) -> None:
        """Move the worldbody place-zone site for predefined tasks."""
        position = np.asarray(position, dtype=np.float32).copy()
        if position[2] < self.table_surface_z:
            position[2] = self.table_surface_z + 0.002

        if self.place_zone_site_id >= 0:
            self.model.site_pos[self.place_zone_site_id] = position

        geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "place_zone_visual"
        )
        if geom_id >= 0:
            self.model.geom_pos[geom_id] = position + np.array([0.0, 0.0, -0.002])

    def _activate_grasp(self, obj_idx: int):
        """Mark a verified grasp/contact and set how the object is carried."""
        self.is_grasped = True
        self.grasp_triggered = True
        self.no_contact_grace_steps = 0
        if self.grasp_carry_offset is not None:
            # Snap the object to the fingertip (real contact-pickup look) rather
            # than freezing the (looser) proximity offset.
            self.grasp_local_offset = np.array(
                [0.0, 0.0, -float(self.grasp_carry_offset)], dtype=np.float32
            )
        else:
            self.grasp_local_offset = (
                self._get_object_position(obj_idx) - self._get_tip_position()
            ).astype(np.float32)
        logger.debug("Grasp attained on object %s", obj_idx)

    def _deactivate_grasp(self, obj_idx: int):
        """Disable programmatic grasp."""
        self.is_grasped = False
        self.grasp_contact_count = 0
        self.no_contact_grace_steps = 0
        logger.debug("Grasp released on object %s", obj_idx)

    def _deactivate_all_grasps(self):
        """Deactivate all programmatic grasp states."""
        self.is_grasped = False
        self.grasp_proximity_count = 0
        self.grasp_contact_count = 0
        self.no_contact_grace_steps = 0
        self._last_tip_object_contact_force = 0.0

    def _tip_object_contact_force(self, obj_idx: int) -> float:
        """Return normal contact force between the tip collision geom and object."""
        if self.tip_geom_id < 0 or obj_idx >= len(self.object_geom_ids):
            return 0.0

        object_geom_id = self.object_geom_ids[obj_idx]
        if object_geom_id < 0:
            return 0.0

        total_force = 0.0
        force = np.zeros(6, dtype=np.float64)
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom_pair = {int(contact.geom1), int(contact.geom2)}
            if self.tip_geom_id in geom_pair and object_geom_id in geom_pair:
                mujoco.mj_contactForce(self.model, self.data, contact_idx, force)
                total_force += max(0.0, float(force[0]))

        return total_force

    def _get_current_raw_obs(self) -> np.ndarray:
        """Get raw observation for current state (single frame)."""
        tip_pos = self._get_tip_position()
        obj_pos = self._get_object_position(self.active_object_idx)
        place_pos = self._get_place_zone_position()
        grasp_state = np.array([1.0 if self.is_grasped else 0.0], dtype=np.float32)
        prev_action = self.previous_action.copy()

        parts = [tip_pos, obj_pos, place_pos, grasp_state, prev_action]

        if self.include_relative_observations:
            parts.extend([tip_pos - obj_pos, obj_pos - place_pos])

        if self.include_actuator_lengths_in_obs:
            actuator_lengths = (
                self.data.actuator_length[: self.num_tendon_actuators]
                .copy()
                .astype(np.float32)
            )
            parts.append(actuator_lengths)

        if self.include_object_velocity_in_obs:
            object_velocity = (obj_pos - self.previous_object_position) / max(
                self.time_per_step, 1e-6
            )
            parts.append(object_velocity.astype(np.float32))

        raw_obs = np.concatenate(parts).astype(np.float32)

        # Add observation noise
        if self.add_observation_noise:
            noise = self.np_random.normal(
                loc=0.0, scale=self.observation_noise_scale, size=raw_obs.shape
            )
            raw_obs += noise

        return raw_obs

    def _get_obs(self) -> np.ndarray:
        """Get stacked observation from buffer."""
        assert len(self.obs_buffer) == self.num_frames, "Observation buffer not full!"
        return np.concatenate(list(self.obs_buffer), axis=0).astype(np.float32)

    def _randomize_object_positions(self):
        """Randomize positions of graspable objects on the desk."""
        num_cubes = self._episode_num_objects
        placed_xy: List[np.ndarray] = []

        for i, body_id in enumerate(self.object_body_ids):
            # Find the joint for this body (freejoint)
            jnt_adr = self.model.body_jntadr[body_id]
            if jnt_adr >= 0:
                qpos_adr = self.model.jnt_qposadr[jnt_adr]
                qvel_adr = self.model.jnt_dofadr[jnt_adr]

                if i < num_cubes:
                    xy = self._sample_nonoverlapping_object_xy(placed_xy)
                    placed_xy.append(xy)
                    new_pos = self._object_rest_position(i, xy)
                    self.data.qpos[qpos_adr : qpos_adr + 3] = new_pos
                else:
                    # Park unused objects far from the workspace but above the floor.
                    self.data.qpos[qpos_adr : qpos_adr + 3] = [
                        10.0 + 0.1 * i,
                        10.0,
                        float(self.object_rest_z[i]),
                    ]

                # Reset orientation (quaternion: w, x, y, z)
                self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]
                # Reset velocity
                self.data.qvel[qvel_adr : qvel_adr + 6] = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Reset state
        self._elapsed_steps = 0
        self.is_grasped = False
        self.grasp_proximity_count = 0
        self.grasp_contact_count = 0
        self.no_contact_grace_steps = 0
        self._last_tip_object_contact_force = 0.0
        self.grasp_triggered = False
        self.place_success = False
        self.obs_buffer.clear()

        # Resolve this episode's object count (curriculum) + reset metric flags.
        self._episode_num_objects = self._current_num_objects()
        self._episode_collision = False
        self._episode_occluded = False

        # Deactivate all grasp constraints
        self._deactivate_all_grasps()

        # Randomize dynamics
        if self.randomize_dynamics:
            self._randomize_dynamic_parameters()

        task = self.task_config
        if task and task.place_position is not None:
            self._set_place_zone_position(
                np.array(task.place_position, dtype=np.float32)
            )

        # Select object to pick (only from the spawned-on-desk subset)
        num_on_desk = self._episode_num_objects
        if task and task.active_object_index is not None:
            self.active_object_idx = int(
                np.clip(task.active_object_index, 0, num_on_desk - 1)
            )
        else:
            self.active_object_idx = self.np_random.integers(0, num_on_desk)

        # Randomize or set object positions
        if task is None or task.randomize_object_position:
            self._randomize_object_positions()
        if task and task.object_position is not None:
            self._set_object_position(
                self.active_object_idx,
                np.array(task.object_position, dtype=np.float32),
            )

        # Randomize initial actuator positions
        if (
            isinstance(self.initial_actuator_config, tuple)
            and len(self.initial_actuator_config) == 2
        ):
            min_val, max_val = self.initial_actuator_config
            initial_positions = self.np_random.uniform(
                low=min_val, high=max_val, size=3
            ).astype(np.float32)
        elif isinstance(self.initial_actuator_config, (float, int)):
            initial_positions = np.full(
                3, float(self.initial_actuator_config), dtype=np.float32
            )
        else:
            initial_positions = np.full(3, 0.23, dtype=np.float32)

        initial_positions = np.clip(
            initial_positions, self.actuator_low[:3], self.actuator_high[:3]
        )

        self.calibrated_tendon_lengths = {
            name: initial_positions[i] for i, name in enumerate(MOTOR_NAMES)
        }

        self.current_position = initial_positions.copy()
        self.data.ctrl[: self.num_tendon_actuators] = self.current_position
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self.previous_previous_action = np.zeros(self.action_dim, dtype=np.float32)

        mujoco.mj_forward(self.model, self.data)

        # Set target site to object position for visualization
        obj_pos = self._get_object_position(self.active_object_idx)
        if self.target_site_id >= 0:
            self.data.site_xpos[self.target_site_id] = obj_pos.copy()

        mujoco.mj_forward(self.model, self.data)
        self.previous_object_position = self._get_object_position(
            self.active_object_idx
        ).copy()

        # Target-clutter / occlusion proxy for this episode's spawn layout.
        self._episode_occluded = self._compute_occlusion()

        # Fill observation buffer
        raw_obs = self._get_current_raw_obs()
        for _ in range(self.num_frames):
            self.obs_buffer.append(raw_obs)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raw_action = np.clip(action, self.action_space.low, self.action_space.high)
        previous_action = self.previous_action.copy()

        # Smooth and rate-limit actions to better match the physical controller.
        action = (
            self.action_smoothing_alpha * raw_action
            + (1.0 - self.action_smoothing_alpha) * previous_action
        ).astype(np.float32)
        if self.max_action_delta_per_step is not None:
            delta = action - previous_action
            delta_norm = np.linalg.norm(delta)
            if delta_norm > self.max_action_delta_per_step > 0:
                action = (
                    previous_action
                    + delta / delta_norm * self.max_action_delta_per_step
                ).astype(np.float32)

        prev_obj_pos = self._get_object_position(self.active_object_idx)
        prev_obj_to_place = np.linalg.norm(
            prev_obj_pos - self._get_place_zone_position()
        )

        # Convert 2D cursor to 3 tendon lengths
        baseline_lengths = np.array(
            [self.calibrated_tendon_lengths[m] for m in MOTOR_NAMES], dtype=np.float32
        )
        new_tendon_lengths = convert_2d_cursor_to_target_lengths(
            action,
            baseline_lengths,
            self.actuator_low,
            self.actuator_high,
            self.max_2d_action_magnitude,
        )

        # Apply controls (tendons only; base-yaw actuator stays at 0 for RL)
        self.data.ctrl[: self.num_tendon_actuators] = new_tendon_lengths
        self.current_position = new_tendon_lengths.copy()

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        if self.is_grasped and self.assisted_grasp_enabled:
            desired_pos = self._get_tip_position() + self.grasp_local_offset
            current_pos = self._get_object_position(self.active_object_idx)
            assisted_pos = (
                self.assisted_grasp_follow_alpha * desired_pos
                + (1.0 - self.assisted_grasp_follow_alpha) * current_pos
            )
            self._set_object_position(self.active_object_idx, assisted_pos)
            mujoco.mj_forward(self.model, self.data)

        self._elapsed_steps += 1

        # Increment global step counter safely
        if hasattr(self._global_step_counter, "get_lock"):
            with self._global_step_counter.get_lock():
                self._global_step_counter.value += 1
        elif hasattr(self._global_step_counter, "value"):
            self._global_step_counter.value += 1
        else:
            self._global_step_counter[0] += 1

        # Get positions
        tip_pos = self._get_tip_position()
        place_pos = self._get_place_zone_position()

        # Find the closest cube to the tip
        min_dist = float("inf")
        closest_obj_pos = None
        closest_idx = 0

        for i, body_id in enumerate(self.object_body_ids):
            cube_pos = self.data.xpos[body_id]
            # Ignore objects parked outside the workspace.
            if cube_pos[2] < -1.0 or np.linalg.norm(cube_pos[:2]) > 1.0:
                continue
            dist = np.linalg.norm(tip_pos - cube_pos)
            if dist < min_dist:
                min_dist = dist
                closest_obj_pos = cube_pos.copy()
                closest_idx = i

        if closest_obj_pos is None:
            closest_obj_pos = self._get_object_position(self.active_object_idx)
            min_dist = np.linalg.norm(tip_pos - closest_obj_pos)
            closest_idx = self.active_object_idx

        # If not grasped, set active object to closest
        if not self.is_grasped:
            self.active_object_idx = closest_idx

        obj_pos = self._get_object_position(self.active_object_idx)

        # Update target visualization
        if self.target_site_id >= 0:
            if not self.is_grasped:
                self.data.site_xpos[self.target_site_id] = obj_pos.copy()
            else:
                self.data.site_xpos[self.target_site_id] = place_pos.copy()

        mujoco.mj_forward(self.model, self.data)

        # Distances
        tip_to_obj = np.linalg.norm(tip_pos - obj_pos)
        obj_to_place = np.linalg.norm(obj_pos - place_pos)
        tip_to_place = np.linalg.norm(tip_pos - place_pos)
        object_progress = prev_obj_to_place - obj_to_place

        # Get curriculum phase
        phase = self._get_curriculum_phase()

        # ===== GRASPING LOGIC (hybrid trigger: real contact OR close proximity) =====
        was_grasped = self.is_grasped
        was_dropped = False
        contact_force = self._tip_object_contact_force(self.active_object_idx)
        self._last_tip_object_contact_force = contact_force
        has_tip_object_contact = contact_force >= self.grasp_contact_force_threshold
        collision = self._compute_collision()
        self._episode_collision = self._episode_collision or collision
        within_grasp_proximity = tip_to_obj <= self.grasp_distance_threshold
        grasp_signal = has_tip_object_contact or within_grasp_proximity
        lift_height = max(0.0, float(obj_pos[2] - self.object_rest_z[self.active_object_idx]))
        is_lifted = lift_height >= LIFT_SUCCESS_HEIGHT

        # Track real contact separately from the (looser) grasp signal so the
        # detector state is honest and debuggable.
        self.grasp_contact_count = self.grasp_contact_count + 1 if has_tip_object_contact else 0

        if not self.is_grasped:
            if grasp_signal:
                self.grasp_proximity_count += 1
                if (
                    self.grasp_proximity_count >= self.grasp_consecutive_steps
                    and phase >= PickPlacePhase.REACH_GRASP
                ):
                    self._activate_grasp(self.active_object_idx)
            else:
                self.grasp_proximity_count = 0
        else:
            # While carried (assisted) the grasp holds; otherwise it must keep a
            # contact/proximity signal or it is released after a short grace.
            if grasp_signal or self.assisted_grasp_enabled:
                self.no_contact_grace_steps = 0
            else:
                self.no_contact_grace_steps += 1
                if self.no_contact_grace_steps >= self.grasp_release_grace_steps:
                    self._deactivate_grasp(self.active_object_idx)
                    was_dropped = not is_lifted

            if obj_pos[2] < self.table_surface_z - 0.02:
                self._deactivate_grasp(self.active_object_idx)
                was_dropped = True

        obj_on_table_or_above = obj_pos[2] >= self.object_rest_z[self.active_object_idx] - 0.005
        self.place_success = (
            phase >= PickPlacePhase.FULL
            and self.is_grasped
            and obj_to_place <= self.place_distance_threshold
            and obj_on_table_or_above
        )

        # ===== REWARD CALCULATION =====
        reward = -self.reach_reward_scale * tip_to_obj

        if has_tip_object_contact:
            reward += self.stable_contact_bonus_scale * min(contact_force, 5.0)
        if within_grasp_proximity and not self.is_grasped:
            # Reward settling into grasp range (replaces the old proximity penalty).
            reward += self.grasp_proximity_bonus_scale * (
                1.0 - tip_to_obj / max(self.grasp_distance_threshold, 1e-6)
            )

        if self.is_grasped and not was_grasped:
            reward += self.grasp_bonus

        if self.is_grasped or has_tip_object_contact:
            # Potential-based (signed) progress: moving toward the zone is
            # rewarded, away is penalised, so oscillation nets zero.
            reward += self.object_progress_reward_scale * object_progress
            reward += self.transport_reward_scale * lift_height

        if was_dropped:
            reward -= self.drop_penalty

        if self.place_success:
            reward += self.place_bonus

        # Per-step living cost: makes finishing (placing) preferable to hovering.
        reward -= self.time_penalty

        # Action change penalty
        action_change = np.linalg.norm(action - previous_action)
        reward -= self.action_change_penalty_scale * action_change
        action_jerk = np.linalg.norm(
            action - 2.0 * previous_action + self.previous_previous_action
        )
        reward -= self.action_jerk_penalty_scale * action_jerk

        # Termination / truncation
        terminated = self.place_success  # Succeed = terminate early
        truncated = self._elapsed_steps >= self._max_episode_steps

        # Update observation
        raw_obs = self._get_current_raw_obs()
        self.obs_buffer.append(raw_obs)
        observation = self._get_obs()
        self.previous_previous_action = previous_action.copy()
        self.previous_action = action.copy()
        self.previous_object_position = obj_pos.copy()

        # Build info
        info = self._get_info()
        info["tip_to_object_distance"] = tip_to_obj
        info["object_to_place_distance"] = obj_to_place
        info["tip_to_place_distance"] = tip_to_place
        info["object_progress"] = object_progress
        info["tip_object_contact"] = has_tip_object_contact
        info["tip_object_contact_force"] = contact_force
        info["grasp_within_proximity"] = within_grasp_proximity
        info["grasp_signal"] = grasp_signal
        info["object_lift_height"] = lift_height
        info["object_lifted"] = is_lifted
        info["is_grasped"] = self.is_grasped
        info["was_dropped"] = was_dropped
        info["place_success"] = self.place_success
        # Multi-object curriculum metrics
        info["num_objects"] = self._episode_num_objects
        info["collision"] = collision
        info["episode_collision"] = self._episode_collision
        info["occluded"] = self._episode_occluded
        info["curriculum_phase"] = phase
        info["active_object_idx"] = self.active_object_idx
        info["tip_position"] = tip_pos.copy()
        info["object_position"] = obj_pos.copy()
        info["place_position"] = place_pos.copy()
        info["raw_action"] = raw_action.copy()
        info["smoothed_action"] = action.copy()
        info["action_change"] = action_change
        info["action_jerk"] = action_jerk

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        if hasattr(self._global_step_counter, "value"):
            total_steps = self._global_step_counter.value
        else:
            total_steps = self._global_step_counter[0]

        return {
            "elapsed_steps": self._elapsed_steps,
            "global_steps": total_steps,
        }

    def render(self) -> Union[List[RenderFrame], RenderFrame, None]:
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                raise RuntimeError("Renderer not initialized for rgb_array mode.")
            self.renderer.update_scene(self.data, camera="fixed_overview")
            return self.renderer.render()
        elif self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    # ===== Domain Randomization =====
    def _store_original_dynamics(self):
        """Store original dynamics parameters."""
        self._original_dynamics = {}
        for param_name in self.randomization_factors:
            if hasattr(self.model, param_name):
                self._original_dynamics[param_name] = getattr(
                    self.model, param_name
                ).copy()

    def _restore_original_dynamics(self):
        """Restore original dynamics parameters."""
        if self._original_dynamics is None:
            return
        for param_name, values in self._original_dynamics.items():
            if hasattr(self.model, param_name):
                getattr(self.model, param_name)[:] = values

    def _randomize_dynamic_parameters(self):
        """Randomize dynamics parameters for sim2real transfer."""
        if self._original_dynamics is None:
            self._store_original_dynamics()
            if self._original_dynamics is None:
                return

        self._restore_original_dynamics()

        for param_name, factor in self.randomization_factors.items():
            if factor <= 0:
                continue
            if (
                hasattr(self.model, param_name)
                and param_name in self._original_dynamics
            ):
                original = self._original_dynamics[param_name]
                param_array = getattr(self.model, param_name)
                multipliers = self.np_random.uniform(
                    1.0 - factor, 1.0 + factor, size=original.shape
                )
                new_values = original * multipliers
                if param_name in [
                    "body_mass",
                    "dof_damping",
                    "geom_friction",
                    "jnt_stiffness",
                ]:
                    new_values = np.maximum(new_values, 1e-9)
                elif param_name == "body_inertia":
                    new_values = np.maximum(new_values, 1e-9)
                param_array[:] = new_values
