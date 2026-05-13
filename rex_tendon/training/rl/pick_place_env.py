"""MuJoCo-based pick-and-place RL environment for tendon-driven continuum robots."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import mujoco
import mujoco.viewer
import os
from collections import deque
import logging

from ...common.constants import MOTOR_NAMES
from ...configs.pick_place_config import PickPlaceEnvConfig
from ...control.geometry import convert_2d_cursor_to_target_lengths

logger = logging.getLogger(__name__)


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
        render_mode: str = None,
        global_step_counter: Optional[list] = None,
        task_config=None,
    ):
        super().__init__()

        self.config = config or PickPlaceEnvConfig()
        self.render_mode = render_mode
        self.task_config = task_config

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

        # Time parameters
        self.simulation_length_seconds = self.config.simulation_length_seconds
        self.time_between_steps_seconds = self.config.time_between_steps_seconds
        self.timestep = self.model.opt.timestep
        self.frame_skip = max(1, round(self.time_between_steps_seconds / self.timestep))
        self.time_per_step = self.frame_skip * self.timestep
        self._max_episode_steps = int(self.simulation_length_seconds / self.time_per_step)

        # Config shortcuts
        self.initial_actuator_config = self.config.initial_actuator_position
        self.tip_site_name = self.config.tip_site_name
        self.num_frames = self.config.num_frames
        self.obs_buffer = deque(maxlen=self.num_frames)
        self.include_actuator_lengths_in_obs = self.config.include_actuator_lengths_in_obs
        self.include_relative_observations = self.config.include_relative_observations
        self.include_object_velocity_in_obs = self.config.include_object_velocity_in_obs
        self.max_2d_action_magnitude = self.config.max_2d_action_magnitude
        self.action_smoothing_alpha = np.clip(self.config.action_smoothing_alpha, 0.0, 1.0)
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
        self.action_change_penalty_scale = self.config.action_change_penalty_scale
        self.action_jerk_penalty_scale = self.config.action_jerk_penalty_scale
        self.object_progress_reward_scale = self.config.object_progress_reward_scale
        self.stable_contact_bonus_scale = self.config.stable_contact_bonus_scale
        self.assisted_grasp_enabled = self.config.assisted_grasp_enabled
        self.assisted_grasp_follow_alpha = np.clip(
            self.config.assisted_grasp_follow_alpha, 0.0, 1.0
        )

        # Grasping parameters
        self.grasp_distance_threshold = self.config.grasp_distance_threshold
        self.grasp_consecutive_steps = self.config.grasp_consecutive_steps
        self.place_distance_threshold = self.config.place_distance_threshold

        # Curriculum
        self.curriculum_enabled = self.config.curriculum_enabled
        self.reach_only_steps = self.config.reach_only_steps
        self.reach_grasp_steps = self.config.reach_grasp_steps

        # Object spawn bounds
        self.object_spawn_min = np.array(self.config.object_spawn_bounds_min)
        self.object_spawn_max = np.array(self.config.object_spawn_bounds_max)

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

        for name in self.config.object_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid == -1:
                raise ValueError(f"Body '{name}' not found in model.")
            self.object_body_ids.append(bid)

        for name in self.config.object_site_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            self.object_site_ids.append(sid)

        for name in self.config.object_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.object_geom_ids.append(gid)

        # Equality constraints removed for programmatic grasping
        self.grasp_constraint_ids = [-1] * len(self.config.object_names)

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
        actuator_ctrlrange = self.model.actuator_ctrlrange
        self.actuator_low = actuator_ctrlrange[:, 0]
        self.actuator_high = actuator_ctrlrange[:, 1]

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

    def _get_curriculum_phase(self) -> int:
        """Determine current curriculum phase based on global step count."""
        if not self.curriculum_enabled:
            return PickPlacePhase.FULL
            
        # Handle both list and multiprocessing.Value types
        if hasattr(self._global_step_counter, 'value'):
            total_steps = self._global_step_counter.value
        else:
            total_steps = self._global_step_counter[0]
            
        if total_steps < self.reach_only_steps:
            return PickPlacePhase.REACH
        elif total_steps < self.reach_grasp_steps:
            return PickPlacePhase.REACH_GRASP
        else:
            return PickPlacePhase.FULL

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

        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        qvel_adr = self.model.jnt_dofadr[jnt_adr]
        self.data.qpos[qpos_adr:qpos_adr + 3] = position
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0

    def _set_place_zone_position(self, position: np.ndarray) -> None:
        """Move the worldbody place-zone site for predefined tasks."""
        if self.place_zone_site_id >= 0:
            self.model.site_pos[self.place_zone_site_id] = position

        geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "place_zone_visual"
        )
        if geom_id >= 0:
            self.model.geom_pos[geom_id] = position + np.array([0.0, 0.0, -0.002])

    def _activate_grasp(self, obj_idx: int):
        """Enable kinematic programmatic grasp flag (physics teleport removed)."""
        self.is_grasped = True
        self.grasp_triggered = True
        self.grasp_local_offset = (
            self._get_object_position(obj_idx) - self._get_tip_position()
        ).astype(np.float32)
        logger.debug(f"Grasp proximity attained on object {obj_idx}")

    def _deactivate_grasp(self, obj_idx: int):
        """Disable programmatic grasp."""
        self.is_grasped = False
        logger.debug(f"Grasp released on object {obj_idx}")

    def _deactivate_all_grasps(self):
        """Deactivate all programmatic grasp states."""
        self.is_grasped = False
        self.grasp_proximity_count = 0

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
            actuator_lengths = self.data.actuator_length.copy().astype(np.float32)
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
        for i, body_id in enumerate(self.object_body_ids):
            # Find the joint for this body (freejoint)
            jnt_adr = self.model.body_jntadr[body_id]
            if jnt_adr >= 0:
                qpos_adr = self.model.jnt_qposadr[jnt_adr]
                # Random XY position on desk, Z at desk surface
                new_pos = self.np_random.uniform(
                    low=self.object_spawn_min,
                    high=self.object_spawn_max,
                )
                # Spread objects apart by adding offset per object
                offset = np.array([0.08 * (i - 1), 0.0, 0.0])
                new_pos = new_pos + offset
                new_pos = np.clip(new_pos, self.object_spawn_min, self.object_spawn_max)

                self.data.qpos[qpos_adr:qpos_adr + 3] = new_pos
                # Reset orientation (quaternion: w, x, y, z)
                self.data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
                # Reset velocity
                qvel_adr = self.model.jnt_dofadr[jnt_adr]
                self.data.qvel[qvel_adr:qvel_adr + 6] = 0

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
        self.grasp_triggered = False
        self.place_success = False
        self.obs_buffer.clear()

        # Deactivate all grasp constraints
        self._deactivate_all_grasps()

        # Randomize dynamics
        if self.randomize_dynamics:
            self._randomize_dynamic_parameters()

        task = self.task_config
        if task and task.place_position is not None:
            self._set_place_zone_position(np.array(task.place_position, dtype=np.float32))

        # Select object to pick
        if task and task.active_object_index is not None:
            self.active_object_idx = int(
                np.clip(task.active_object_index, 0, len(self.object_body_ids) - 1)
            )
        else:
            self.active_object_idx = self.np_random.integers(0, len(self.object_body_ids))

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
            initial_positions = np.full(3, float(self.initial_actuator_config), dtype=np.float32)
        else:
            initial_positions = np.full(3, 0.23, dtype=np.float32)

        initial_positions = np.clip(
            initial_positions, self.actuator_low[:3], self.actuator_high[:3]
        )

        self.calibrated_tendon_lengths = {
            name: initial_positions[i] for i, name in enumerate(MOTOR_NAMES)
        }

        self.current_position = initial_positions.copy()
        self.data.ctrl[:] = self.current_position
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self.previous_previous_action = np.zeros(self.action_dim, dtype=np.float32)

        # Set target site to object position for visualization
        obj_pos = self._get_object_position(self.active_object_idx)
        if self.target_site_id >= 0:
            self.data.site_xpos[self.target_site_id] = obj_pos.copy()

        mujoco.mj_forward(self.model, self.data)
        self.previous_object_position = self._get_object_position(
            self.active_object_idx
        ).copy()

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

        # Apply controls
        self.data.ctrl[:] = new_tendon_lengths
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
        if hasattr(self._global_step_counter, 'get_lock'):
            with self._global_step_counter.get_lock():
                self._global_step_counter.value += 1
        elif hasattr(self._global_step_counter, 'value'):
            self._global_step_counter.value += 1
        else:
            self._global_step_counter[0] += 1

        # Get positions
        tip_pos = self._get_tip_position()
        obj_pos = self._get_object_position(self.active_object_idx)
        place_pos = self._get_place_zone_position()

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

        # ===== GRASPING LOGIC =====
        was_grasped = self.is_grasped
        was_dropped = False

        if not self.is_grasped:
            # Check if tip is close enough to object
            if tip_to_obj < self.grasp_distance_threshold:
                self.grasp_proximity_count += 1
                if (
                    self.grasp_proximity_count >= self.grasp_consecutive_steps
                    and phase >= PickPlacePhase.REACH_GRASP
                ):
                    self._activate_grasp(self.active_object_idx)
            else:
                self.grasp_proximity_count = 0
        else:
            # Check for drop (object too far from tip while grasped)
            if tip_to_obj > self.grasp_distance_threshold * 3:
                self._deactivate_grasp(self.active_object_idx)
                was_dropped = True

        # ===== REWARD CALCULATION =====
        reward = 0.0

        if phase == PickPlacePhase.REACH:
            # Phase 1: Reach only
            reward = -self.reach_reward_scale * tip_to_obj

        elif phase == PickPlacePhase.REACH_GRASP:
            if not self.is_grasped:
                # Reach toward object
                reward = -self.reach_reward_scale * tip_to_obj
                # Bonus for being very close
                if tip_to_obj < self.grasp_distance_threshold:
                    reward += 0.5
            else:
                # Just grasped bonus
                if not was_grasped:
                    reward += self.grasp_bonus
                # Small positive reward for holding
                reward += 0.1

        else:  # FULL phase
            if not self.is_grasped and not self.place_success:
                # Reach toward object
                reward = -self.reach_reward_scale * tip_to_obj
                if tip_to_obj < self.grasp_distance_threshold:
                    reward += 0.5
                # Penalty if we dropped it
                if was_grasped:
                    reward -= self.drop_penalty
            elif self.is_grasped:
                # Just grasped bonus
                if not was_grasped:
                    reward += self.grasp_bonus
                # Transport to place zone
                reward -= self.transport_reward_scale * obj_to_place
                reward += self.object_progress_reward_scale * object_progress
                # Place success check
                if obj_to_place < self.place_distance_threshold:
                    self.place_success = True
                    reward += self.place_bonus
                    # Release the object
                    self._deactivate_grasp(self.active_object_idx)
            elif self.place_success:
                # Object placed, small positive reward
                reward += 0.5

        # Action change penalty
        action_change = np.linalg.norm(action - previous_action)
        reward -= self.action_change_penalty_scale * action_change
        action_jerk = np.linalg.norm(
            action - 2.0 * previous_action + self.previous_previous_action
        )
        reward -= self.action_jerk_penalty_scale * action_jerk

        if tip_to_obj < self.grasp_distance_threshold:
            contact_quality = 1.0 - tip_to_obj / max(self.grasp_distance_threshold, 1e-6)
            reward += self.stable_contact_bonus_scale * contact_quality

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
        info["is_grasped"] = self.is_grasped
        info["was_dropped"] = was_dropped
        info["place_success"] = self.place_success
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

        return observation, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        if hasattr(self._global_step_counter, 'value'):
            total_steps = self._global_step_counter.value
        else:
            total_steps = self._global_step_counter[0]
            
        return {
            "elapsed_steps": self._elapsed_steps,
            "global_steps": total_steps,
        }

    def render(self) -> Optional[Union[np.ndarray, None]]:
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
                self._original_dynamics[param_name] = getattr(self.model, param_name).copy()

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
            if hasattr(self.model, param_name) and param_name in self._original_dynamics:
                original = self._original_dynamics[param_name]
                param_array = getattr(self.model, param_name)
                multipliers = self.np_random.uniform(
                    1.0 - factor, 1.0 + factor, size=original.shape
                )
                new_values = original * multipliers
                if param_name in ["body_mass", "dof_damping", "geom_friction", "jnt_stiffness"]:
                    new_values = np.maximum(new_values, 1e-9)
                elif param_name == "body_inertia":
                    new_values = np.maximum(new_values, 1e-9)
                param_array[:] = new_values

