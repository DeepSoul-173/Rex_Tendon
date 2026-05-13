"""MuJoCo-based reinforcement learning environment for tentacle robots."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import mujoco.viewer
import os
from scipy.interpolate import CubicSpline
from collections import deque
import logging

from ...common.constants import MOTOR_NAMES
from ...configs.rl_training import RLEnvironmentConfig
from ...control.geometry import convert_2d_cursor_to_target_lengths

logger = logging.getLogger(__name__)


class TentacleTargetFollowingEnv(gym.Env):
    """Gymnasium environment for controlling a MuJoCo tentacle to follow a smooth, randomly generated target trajectory.
    The observation consists of the tip position, target position, and the previous action.
    The reward is based on the distance between the tip and the target, penalized by action magnitude.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        config: Optional[RLEnvironmentConfig] = None,
        render_mode: str = None,
    ):
        super().__init__()

        self.config = config or RLEnvironmentConfig()
        self.render_mode = render_mode

        # Extract config parameters for easier access
        xml_file = self.config.xml_file
        if not os.path.exists(xml_file):
            script_dir = os.path.dirname(__file__)
            abs_xml_file = os.path.join(script_dir, xml_file)
            if not os.path.exists(abs_xml_file):
                raise FileNotFoundError(
                    f"Cannot find MuJoCo XML file at {xml_file} or {abs_xml_file}"
                )
            xml_file = abs_xml_file

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # Time-based parameters from config
        self.simulation_length_seconds = self.config.simulation_length_seconds
        self.time_between_steps_seconds = self.config.time_between_steps_seconds
        self.timestep = self.model.opt.timestep

        # Calculate frame_skip based on desired time between steps
        self.frame_skip = max(1, round(self.time_between_steps_seconds / self.timestep))
        self.time_per_step = self.frame_skip * self.timestep

        # Calculate max_episode_steps based on simulation length and actual time per step
        self._max_episode_steps = int(
            self.simulation_length_seconds / self.time_per_step
        )

        # Extract config parameters
        self.initial_actuator_config = self.config.initial_actuator_position
        self.camera_names = [
            "fixed_overview",
            "tracking_cam",
        ]
        self.image_size = (84, 84)
        self.tip_site_name = self.config.tip_site_name
        self.target_bounds_min = np.array(self.config.target_bounds_min)
        self.target_bounds_max = np.array(self.config.target_bounds_max)
        self.reward_distance_scale = self.config.reward_distance_scale
        self.render_mode = render_mode
        self._elapsed_steps = 0
        self.current_position = None
        self.spline_x = None
        self.spline_y = None
        self.spline_z = None
        self.trajectory_segments = []
        self.num_frames = self.config.num_frames
        self.obs_buffer = deque(maxlen=self.num_frames)
        self.include_actuator_lengths_in_obs = (
            self.config.include_actuator_lengths_in_obs
        )

        # Sim2Real Additions
        self.randomize_dynamics = self.config.randomize_dynamics
        self.randomization_factors = self.config.randomization_factors
        self.add_observation_noise = self.config.add_observation_noise
        self.observation_noise_scale = self.config.observation_noise_scale
        self._store_original_dynamics()  # Store original params if randomizing

        # Trajectory Modification
        self.pause_probability = self.config.pause_probability
        self.min_pause_duration = self.config.min_pause_duration
        self.max_pause_duration = self.config.max_pause_duration
        self.min_move_duration = self.config.min_move_duration
        self.max_move_duration = self.config.max_move_duration

        # Penalties
        self.action_change_penalty_scale = self.config.action_change_penalty_scale
        self.distance_penalty_exponent = self.config.distance_penalty_exponent

        # 2D Action Space Setup
        self.max_2d_action_magnitude = self.config.max_2d_action_magnitude

        # Calibrated tendon lengths will be set in reset()
        self.calibrated_tendon_lengths = {name: 0.0 for name in MOTOR_NAMES}

        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(
                self.model, height=self.image_size[0], width=self.image_size[1]
            )
        else:
            self.renderer = None
        self.viewer = None

        actuator_ctrlrange = self.model.actuator_ctrlrange
        low = actuator_ctrlrange[:, 0]
        high = actuator_ctrlrange[:, 1]

        # 2D action space: [x, y] cursor position in [-1, 1] range -> converted to tendon lengths
        self.action_space = spaces.Box(
            low=-self.max_2d_action_magnitude,
            high=self.max_2d_action_magnitude,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_dim = 2

        # Store the original control range for motor position mapping
        self.actuator_low = low
        self.actuator_high = high
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self.previous_tendon_lengths = np.zeros(3, dtype=np.float32)

        # Observation space: tip_position (3) + target_position (3) + actuator_lengths
        single_frame_obs_dim = 6  # Tip pos (3) + Target pos (3)
        if self.include_actuator_lengths_in_obs:
            single_frame_obs_dim += 3

        stacked_obs_shape = (self.num_frames * single_frame_obs_dim,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=stacked_obs_shape, dtype=np.float32
        )

        self.tip_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.tip_site_name
        )
        if self.tip_site_id == -1:
            raise ValueError(f"Site '{self.tip_site_name}' not found in the model.")

        self.target_site_id = -1
        if self.model is not None and self.model.nsite > 0:
            self.target_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
            )
            if self.target_site_id == -1:
                logger.warning(
                    "Site 'target' not found in the model. Will not be visualized in MuJoCo"
                )
        else:
            logger.warning("No sites found in model, target site cannot be identified")

        self.target_position: np.ndarray = np.zeros(3)
        self.tip_history = []  # Store tip history for rendering trajectory
        self.trajectory_segments = []  # Stores final spline/pause segments

        self.fixed_target_position = (
            np.array(self.config.fixed_target_position)
            if self.config.fixed_target_position
            else None
        )

    def _get_obs(self) -> np.ndarray:
        """Retrieves the stacked observation from the buffer."""
        assert len(self.obs_buffer) == self.num_frames, "Observation buffer not full!"
        return np.concatenate(list(self.obs_buffer), axis=0).astype(np.float32)

    def _get_current_raw_obs(self) -> np.ndarray:
        """Gets the raw observation for the current state (single frame)."""
        # Get tip and target positions
        tip_position = self._get_tip_position()
        target_position = self.target_position.copy()

        # Concatenate tip and target positions (cable lengths removed)
        raw_observation_parts = [tip_position, target_position]

        if self.include_actuator_lengths_in_obs:
            actuator_lengths = self.data.actuator_length.copy().astype(np.float32)
            raw_observation_parts.append(actuator_lengths)

        raw_observation = np.concatenate(raw_observation_parts).astype(np.float32)

        # Sim2Real: Add Observation Noise
        if self.add_observation_noise:
            noise = self.np_random.normal(
                loc=0.0, scale=self.observation_noise_scale, size=raw_observation.shape
            )
            raw_observation += noise

        return raw_observation

    def _get_tip_position(self) -> np.ndarray:
        return self.data.site_xpos[self.tip_site_id].copy()

    def _generate_target_trajectory(self):
        """Generates a trajectory as a sequence of move and pause segments."""
        self.trajectory_segments = []

        if self.fixed_target_position is not None:
            segment = {
                "type": "pause",
                "start_time": 0.0,
                "end_time": self.simulation_length_seconds,
                "position": self.fixed_target_position.copy(),
            }
            self.trajectory_segments.append(segment)
            return

        current_time = 0.0

        # Generate random initial position
        current_pos = self.np_random.uniform(
            low=self.target_bounds_min,
            high=self.target_bounds_max,
            size=3,
        )

        is_first_segment = True
        waypoints = [
            (current_time, current_pos.copy())
        ]  # Store (time, position) tuples
        segment_markers = []  # Store {'type': 'move'/'pause', 'end_time': t}

        while current_time < self.simulation_length_seconds:
            remaining_time = self.simulation_length_seconds - current_time
            segment_type = "move"  # Default to move

            # Decide segment type (allow pause only after the first segment)
            if (
                not is_first_segment
                and self.np_random.random() < self.pause_probability
            ):
                segment_type = "pause"

            if segment_type == "pause":
                duration = self.np_random.uniform(
                    self.min_pause_duration, self.max_pause_duration
                )
                duration = min(duration, remaining_time)  # Clamp to remaining time
                end_time = current_time + duration
                # Mark the end of the pause
                segment_markers.append(
                    {
                        "type": "pause",
                        "start_time": current_time,
                        "end_time": end_time,
                        "position": current_pos.copy(),
                    }
                )
                # Add waypoint at the end of the pause (position doesn't change)
                waypoints.append((end_time, current_pos.copy()))
                current_time = end_time
            else:  # Move segment
                duration = self.np_random.uniform(
                    self.min_move_duration, self.max_move_duration
                )
                duration = min(duration, remaining_time)  # Clamp to remaining time
                if duration < 1e-6:  # Avoid tiny segments at the end
                    break
                end_time = current_time + duration
                start_pos = current_pos.copy()  # Keep for print statement
                next_pos = self.np_random.uniform(
                    low=self.target_bounds_min,
                    high=self.target_bounds_max,
                    size=3,
                )
                # Mark the end of the move segment (start time implicitly defined by previous)
                segment_markers.append(
                    {"type": "move", "start_time": current_time, "end_time": end_time}
                )
                # Add the next waypoint
                waypoints.append((end_time, next_pos.copy()))
                current_pos = next_pos  # Update current position for next segment
                current_time = end_time

            is_first_segment = False

        # Post-process waypoints to create smooth splines between pauses
        self.trajectory_segments = []
        current_move_sequence_indices = []  # Indices into the waypoints list

        # Ensure the first waypoint is included if the first segment is a move
        if segment_markers and segment_markers[0]["type"] == "move":
            current_move_sequence_indices.append(0)

        for i, marker in enumerate(segment_markers):
            waypoint_index = (
                i + 1
            )  # Waypoint corresponding to the *end* of this marker's time

            if marker["type"] == "move":
                # Add the index of the waypoint at the *end* of this move segment
                if not current_move_sequence_indices:
                    # Start of a new move sequence, add the *start* waypoint index
                    current_move_sequence_indices.append(i)
                current_move_sequence_indices.append(waypoint_index)

            elif marker["type"] == "pause":
                # Process any preceding move sequence
                if len(current_move_sequence_indices) >= 2:
                    sequence_times = [
                        waypoints[idx][0] for idx in current_move_sequence_indices
                    ]
                    sequence_pos = [
                        waypoints[idx][1] for idx in current_move_sequence_indices
                    ]
                    # Use 'not-a-knot' for smoother internal points if possible
                    bc_type = (
                        "not-a-knot"
                        if len(current_move_sequence_indices) >= 4
                        else "natural"
                    )
                    try:
                        spline = CubicSpline(
                            sequence_times, sequence_pos, axis=0, bc_type=bc_type
                        )
                        move_segment = {
                            "type": "move",
                            "start_time": sequence_times[0],
                            "end_time": sequence_times[-1],
                            "spline": spline,
                        }
                        self.trajectory_segments.append(move_segment)
                    except ValueError as e:

                        # Fallback
                        self.trajectory_segments.append(
                            {
                                "type": "pause",
                                "start_time": sequence_times[0],
                                "end_time": sequence_times[-1],
                                "position": sequence_pos[0],  # Pause at start
                            }
                        )

                # Reset for the next potential move sequence
                current_move_sequence_indices = []

                # Add the pause segment itself
                self.trajectory_segments.append(marker)  # Add the original pause marker

        # Process any trailing move sequence after the last pause (or if only moves)
        if len(current_move_sequence_indices) >= 2:
            sequence_times = [
                waypoints[idx][0] for idx in current_move_sequence_indices
            ]
            sequence_pos = [waypoints[idx][1] for idx in current_move_sequence_indices]
            bc_type = (
                "not-a-knot" if len(current_move_sequence_indices) >= 4 else "natural"
            )
            try:
                spline = CubicSpline(
                    sequence_times, sequence_pos, axis=0, bc_type=bc_type
                )
                move_segment = {
                    "type": "move",
                    "start_time": sequence_times[0],
                    "end_time": sequence_times[-1],
                    "spline": spline,
                }
                self.trajectory_segments.append(move_segment)
            except ValueError as e:
                # Fallback
                self.trajectory_segments.append(
                    {
                        "type": "pause",
                        "start_time": sequence_times[0],
                        "end_time": sequence_times[-1],
                        "position": sequence_pos[0],  # Pause at start
                    }
                )

        # Ensure there's at least one segment if the loop finishes early or only one point
        if not self.trajectory_segments:
            segment = {
                "type": "pause",
                "start_time": 0.0,
                "end_time": self.simulation_length_seconds,
                "position": waypoints[0][1],  # Use initial position
            }
            self.trajectory_segments.append(segment)
        else:
            # Sort segments by start time just in case
            self.trajectory_segments.sort(key=lambda s: s["start_time"])

            # Ensure the last segment reaches the full simulation time if slightly short
            last_segment = self.trajectory_segments[-1]
            if (
                last_segment["end_time"] < self.simulation_length_seconds - 1e-6
            ):  # Add tolerance
                # If the last segment is a move, we might need to adjust its spline or clamp time later.
                # If it's a pause, just extending the time is fine.
                # Let's assume clamping in _get_target_point_at_time is sufficient for now.
                last_segment["end_time"] = self.simulation_length_seconds

    def _get_target_point_at_time(self, time: float) -> np.ndarray:
        """Gets the target position by finding the active segment and evaluating it."""
        if not self.trajectory_segments:
            return np.zeros(3)

        # Clamp time to the valid range of the entire trajectory
        time = np.clip(time, 0, self.simulation_length_seconds)

        active_segment = None
        for segment in self.trajectory_segments:
            # Check if time falls within the segment [start, end)
            # Handle the very end time: associate it with the last segment
            if segment["start_time"] <= time < segment["end_time"] or (
                time == self.simulation_length_seconds
                and segment == self.trajectory_segments[-1]
            ):
                active_segment = segment
                break

        if active_segment is None:
            # This might happen due to float precision, default to last segment
            # Ensure we grab the actual last segment after sorting/processing
            active_segment = (
                self.trajectory_segments[-1] if self.trajectory_segments else None
            )
            if active_segment is None:
                return np.zeros(3)

        if active_segment["type"] == "pause":
            target_pos = active_segment["position"]
        elif active_segment["type"] == "move":
            # Clamp the time to the segment's valid range for the spline evaluation
            spline_time = np.clip(
                time, active_segment["start_time"], active_segment["end_time"]
            )
            try:
                target_pos = active_segment["spline"](spline_time)
            except Exception as e:
                # Fallback: return the end position of the segment
                target_pos = active_segment["end_pos"]
        else:
            target_pos = np.zeros(3)

        # Ensure target stays within bounds (spline might slightly overshoot)
        # Although clipping within the generation might be better?
        return np.clip(target_pos, self.target_bounds_min, self.target_bounds_max)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.trajectory_segments = []  # Reset trajectory segments
        self.tip_history = []  # Reset tip history
        self.obs_buffer.clear()  # Clear observation buffer

        # Sim2Real: Randomize Dynamics
        if self.randomize_dynamics:
            self._randomize_dynamic_parameters()

        # Randomize Initial Actuator Positions
        if (
            isinstance(self.initial_actuator_config, tuple)
            and len(self.initial_actuator_config) == 2
        ):
            min_val, max_val = self.initial_actuator_config
            # Sample individual initial positions for each of the 3 motors
            initial_positions_sampled = self.np_random.uniform(
                low=min_val, high=max_val, size=3
            ).astype(np.float32)
        elif isinstance(self.initial_actuator_config, (float, int)):
            initial_positions_sampled = np.full(
                3, float(self.initial_actuator_config), dtype=np.float32
            )
        else:
            raise ValueError(
                "initial_actuator_position must be a float or a tuple (min, max)"
            )

        # Clip to ensure we're within the allowed ctrlrange
        # Ensure actuator_low and actuator_high are correctly shaped (e.g., (3,) for 3 motors)
        actuator_low_expanded = (
            np.tile(
                self.actuator_low,
                (initial_positions_sampled.shape[0] // self.actuator_low.shape[0], 1),
            ).flatten()
            if self.actuator_low.shape[0] < initial_positions_sampled.shape[0]
            else self.actuator_low
        )
        actuator_high_expanded = (
            np.tile(
                self.actuator_high,
                (initial_positions_sampled.shape[0] // self.actuator_high.shape[0], 1),
            ).flatten()
            if self.actuator_high.shape[0] < initial_positions_sampled.shape[0]
            else self.actuator_high
        )

        initial_positions_sampled = np.clip(
            initial_positions_sampled,
            actuator_low_expanded[:3],
            actuator_high_expanded[:3],
        )

        # Update calibrated tendon lengths based on the (potentially randomized) initial state
        self.calibrated_tendon_lengths = {
            name: initial_positions_sampled[i] for i, name in enumerate(MOTOR_NAMES)
        }

        # Generate a new random trajectory for this episode
        self._generate_target_trajectory()

        # Get the initial target position at time 0
        self.target_position = self._get_target_point_at_time(0.0)
        self.data.site_xpos[self.target_site_id] = self.target_position.copy()
        self._elapsed_steps = 0

        # Initialize current tendon lengths based on (potentially randomized) initial values
        self.current_position = initial_positions_sampled.copy()
        # Set initial tendon lengths in MuJoCo data
        self.data.ctrl[:] = self.current_position
        # Initialize previous action for penalty calculation - dimension depends on action space
        self.previous_action = np.zeros(self.action_dim, dtype=np.float32)
        self.previous_tendon_lengths = self.current_position.copy()

        mujoco.mj_forward(self.model, self.data)
        # Get the raw observation for the initial state
        # No need to set previous_action here for the observation itself anymore
        raw_obs = self._get_current_raw_obs()
        # Fill the buffer with the initial observation
        for _ in range(self.num_frames):
            self.obs_buffer.append(raw_obs)

        observation = self._get_obs()  # Get the stacked observation
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Convert 2-D cursor action to 3 tendon lengths using shared utility
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

        previous_tendon_lengths = self.current_position.copy()

        # Apply the new tendon lengths directly
        self.data.ctrl[:] = new_tendon_lengths

        # Update current position for next step
        self.current_position = new_tendon_lengths.copy()

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Increment step counter before calculating observation
        self._elapsed_steps += 1

        # Calculate current simulation time and update target position based on spline
        current_time = self._elapsed_steps * self.time_per_step
        self.target_position = self._get_target_point_at_time(current_time)
        self.data.site_xpos[self.target_site_id] = self.target_position.copy()
        mujoco.mj_forward(
            self.model, self.data
        )  # Forward kinematics to update site positions

        tip_pos = self._get_tip_position()
        distance = np.linalg.norm(tip_pos - self.target_position)
        self.tip_history.append(tip_pos.copy())  # Store tip point

        # Reward Calculation

        # Base distance penalty (allows exponent)
        distance_penalty = self.reward_distance_scale * (
            distance**self.distance_penalty_exponent
        )

        reward = -distance_penalty

        # Action Change Penalty (Using L2 norm of the applied delta vector)
        action_magnitude = np.linalg.norm(new_tendon_lengths - previous_tendon_lengths)
        action_penalty = self.action_change_penalty_scale * action_magnitude
        reward -= action_penalty

        # Constraint Penalty removed

        # Termination and Truncation
        truncated = self._elapsed_steps >= self._max_episode_steps
        terminated = False  # No specific termination condition other than time limit

        # Store the action taken *before* getting the observation for the *next* state
        # This is still needed for the penalty calculation in the *next* step, if that penalty were based on change
        # current_action_for_obs = action.copy() # If obs needed current action (it doesn't)
        action_applied = (
            new_tendon_lengths  # Store the applied tendon lengths for info/penalties
        )

        # Get the raw observation for the *new* state and add it to the buffer
        raw_obs = (
            self._get_current_raw_obs()
        )  # Does not include previous_action anymore
        self.obs_buffer.append(raw_obs)

        # Get the stacked observation from the updated buffer
        observation = self._get_obs()

        # Update previous_action *after* using the current action for penalties etc.
        # This is now only used if a penalty depends on the *previous* step's action explicitly
        # Note: Should we store action or processed_action here? Storing the *intended* action (pre-constraint)
        # might be better for learning consistency if the constraint is hit often.
        # Let's store the original intended action. If penalties need the *applied* action,
        # we can use action_applied variable defined earlier in the step.
        self.previous_action = action.copy()
        self.previous_tendon_lengths = new_tendon_lengths.copy()

        info = self._get_info()
        info["distance"] = distance
        info["distance_to_target"] = distance
        info["target_position"] = self.target_position.copy()
        info["tip_position"] = tip_pos.copy()
        info["distance_penalty"] = distance_penalty
        # Use the magnitude of the *applied* action for info only (penalty already applied)
        info["action_magnitude"] = action_magnitude
        info["action_penalty"] = action_penalty
        # Also add original and processed action to info for debugging
        info["original_action"] = action.copy()
        info["processed_action"] = action_applied.copy()

        # Add 2D action space specific info
        info["2d_cursor_position"] = action.copy()
        info["tendon_lengths"] = self.current_position.copy()

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_info(self) -> Dict[str, Any]:
        return {
            "target_trajectory": self.trajectory_segments,  # Full target path for viz
            "tip_trajectory": self.tip_history,  # Full tip path for viz
            "elapsed_steps": self._elapsed_steps,
        }

    def render(self) -> Optional[Union[np.ndarray, None]]:
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                raise RuntimeError(
                    "Renderer not initialized for rgb_array render mode."
                )
            self.renderer.update_scene(self.data, camera=self.camera_names[0])
            return self.renderer.render()
        elif self.render_mode == "human":
            self.data.site_xpos[self.target_site_id] = self.target_position.copy()
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    # Sim2Real Helper Methods
    def _store_original_dynamics(self):
        """Stores the original dynamics parameters of the model."""
        if not self.randomize_dynamics:
            self._original_dynamics = None
            return

        self._original_dynamics = {
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "dof_damping": self.model.dof_damping.copy(),
            "jnt_stiffness": self.model.jnt_stiffness.copy(),
            "geom_friction": self.model.geom_friction.copy(),
            "geom_solref": self.model.geom_solref.copy(),
            "geom_solimp": self.model.geom_solimp.copy(),
            "actuator_gainprm": self.model.actuator_gainprm.copy(),
            "actuator_biasprm": self.model.actuator_biasprm.copy(),
            # Add other parameters as needed (e.g., tendon stiffness, joint friction)
        }

    def _restore_original_dynamics(self):
        """Restores the original dynamics parameters to the model."""
        if self._original_dynamics is None:
            return  # Nothing to restore if not randomized or not stored

        self.model.body_mass[:] = self._original_dynamics["body_mass"]
        self.model.body_inertia[:] = self._original_dynamics["body_inertia"]
        self.model.dof_damping[:] = self._original_dynamics["dof_damping"]
        self.model.jnt_stiffness[:] = self._original_dynamics["jnt_stiffness"]
        self.model.geom_friction[:] = self._original_dynamics["geom_friction"]
        self.model.geom_solref[:] = self._original_dynamics["geom_solref"]
        self.model.geom_solimp[:] = self._original_dynamics["geom_solimp"]
        self.model.actuator_gainprm[:] = self._original_dynamics["actuator_gainprm"]
        self.model.actuator_biasprm[:] = self._original_dynamics["actuator_biasprm"]
        # Restore others as needed

    def _randomize_dynamic_parameters(self):
        """Randomizes specified dynamic parameters based on randomization_factors."""
        if self._original_dynamics is None:
            self._store_original_dynamics()  # Attempt to store now
            if self._original_dynamics is None:  # Check if storing failed
                return

        # Restore original parameters before applying new randomization
        self._restore_original_dynamics()

        for param_name, factor in self.randomization_factors.items():
            if factor <= 0:
                continue  # Skip if factor is non-positive

            if hasattr(self.model, param_name):
                original_values = self._original_dynamics[param_name]
                param_array = getattr(
                    self.model, param_name
                )  # Get reference to model's array
                random_multipliers = self.np_random.uniform(
                    1.0 - factor, 1.0 + factor, size=original_values.shape
                )
                new_values = original_values * random_multipliers

                # Ensure non-negativity for relevant parameters
                if param_name in [
                    "body_mass",
                    "dof_damping",
                    "geom_friction",
                    "jnt_stiffness",
                ]:
                    # Use maximum with a small epsilon to avoid exactly zero
                    new_values = np.maximum(new_values, 1e-9)
                elif param_name == "body_inertia":
                    # Inertia needs to be positive definite, simple element-wise max is okay for diagonal
                    new_values = np.maximum(new_values, 1e-9)

                param_array[:] = new_values  # Modify the model's array directly

            else:
                pass


def env_creator(env_config: Dict[str, Any]) -> TentacleTargetFollowingEnv:
    """Creator function for RLlib registration."""
    config = RLEnvironmentConfig(**env_config)
    render_mode = env_config.get("render_mode", None)
    return TentacleTargetFollowingEnv(config=config, render_mode=render_mode)

