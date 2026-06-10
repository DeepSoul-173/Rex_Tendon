"""Pick-and-place RL training with PPO and curriculum learning."""

import logging
import multiprocessing as mp
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime
from typing import Optional
import shutil

import yaml
import typer
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from rich.console import Console

from .pick_place_env import TentaclePickPlaceEnv
from ...configs.pick_place_config import PickPlaceConfig, PickPlaceEnvConfig

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="Pick-and-place RL training utilities")


class PickPlaceMetricsCallback(BaseCallback):
    """Log pick-and-place metrics as rolling-window rates over recent episodes.

    Cumulative lifetime rates barely move once millions of steps accumulate, so
    they hide whether the policy is *currently* improving. Rolling windows make
    grasp/lift/place/drop rates responsive to recent performance.
    """

    def __init__(self, verbose: int = 0, window: int = 200):
        super().__init__(verbose)
        self.episode_count = 0
        self.ep_grasped = deque(maxlen=window)
        self.ep_lifted = deque(maxlen=window)
        self.ep_placed = deque(maxlen=window)
        self.ep_dropped = deque(maxlen=window)
        self.action_changes = deque(maxlen=2000)
        self.action_jerks = deque(maxlen=2000)
        self.object_distances = deque(maxlen=2000)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "action_change" in info:
                self.action_changes.append(info["action_change"])
            if "action_jerk" in info:
                self.action_jerks.append(info["action_jerk"])
            if "object_to_place_distance" in info:
                self.object_distances.append(info["object_to_place_distance"])

            # Record one outcome per completed episode (Monitor adds "episode").
            if "episode" in info:
                self.episode_count += 1
                placed = bool(info.get("place_success", False))
                self.ep_placed.append(1.0 if placed else 0.0)
                self.ep_grasped.append(
                    1.0 if (placed or info.get("is_grasped", False)) else 0.0
                )
                self.ep_lifted.append(1.0 if info.get("object_lifted", False) else 0.0)
                self.ep_dropped.append(1.0 if info.get("was_dropped", False) else 0.0)

        # Log rolling metrics every 1000 steps
        if self.num_timesteps % 1000 == 0 and self.ep_placed:
            self.logger.record(
                "pick_place/grasp_success_rate", 100.0 * float(np.mean(self.ep_grasped))
            )
            self.logger.record(
                "pick_place/lift_rate", 100.0 * float(np.mean(self.ep_lifted))
            )
            self.logger.record(
                "pick_place/place_success_rate", 100.0 * float(np.mean(self.ep_placed))
            )
            self.logger.record(
                "pick_place/drop_rate", 100.0 * float(np.mean(self.ep_dropped))
            )
            self.logger.record("pick_place/total_episodes", self.episode_count)
            self.logger.record("pick_place/window_episodes", len(self.ep_placed))
            if self.action_changes:
                self.logger.record(
                    "pick_place/mean_action_change", float(np.mean(self.action_changes))
                )
            if self.action_jerks:
                self.logger.record(
                    "pick_place/mean_action_jerk", float(np.mean(self.action_jerks))
                )
            if self.object_distances:
                self.logger.record(
                    "pick_place/mean_object_to_place_distance",
                    float(np.mean(self.object_distances)),
                )

            # Get curriculum phase from first info
            for info in self.locals.get("infos", []):
                if "curriculum_phase" in info:
                    self.logger.record(
                        "pick_place/curriculum_phase", info["curriculum_phase"]
                    )
                    break

        return True


class SaveVecNormalizeCallback(BaseCallback):
    """Persist VecNormalize statistics during and after training."""

    def __init__(self, save_path: Path, save_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_freq = max(1, save_freq)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            self._save_stats("vecnormalize.pkl")
        return True

    def _on_training_end(self) -> None:
        self._save_stats("vecnormalize_final.pkl")

    def _save_stats(self, filename: str) -> None:
        env = self.model.get_vec_normalize_env()
        if env is not None:
            env.save(str(self.save_path / filename))


class MultiObjectCurriculumCallback(BaseCallback):
    """Ramp the spawned-object count as the policy masters each level, and log the
    per-object-count metrics that form the generalization curve.

    Promotion is adaptive: when the rolling place-success rate at the current
    object count exceeds ``promote_threshold`` (over a full window, after a
    minimum number of episodes at the level), the shared object counter is bumped
    and a per-stage best checkpoint is saved. The final checkpoint at the end of
    training is the multi-object policy.

    TensorBoard tags (under the run's logdir, e.g. rex_results/pick_place_multi/):
      curriculum/place_success_rate/{n}obj
      curriculum/grasp_success_rate/{n}obj
      curriculum/episode_length/{n}obj
      curriculum/collision_rate, curriculum/occlusion_rate, curriculum/num_objects
    """

    def __init__(
        self,
        curriculum_state,
        min_objects: int,
        max_objects: int,
        model_dir: Path,
        promote_threshold: float = 0.90,
        window: int = 100,
        min_episodes_at_level: int = 300,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.state = curriculum_state
        self.min_objects = int(min_objects)
        self.max_objects = int(max_objects)
        self.model_dir = Path(model_dir)
        self.promote_threshold = float(promote_threshold)
        self.window = int(window)
        self.min_episodes_at_level = int(min_episodes_at_level)
        self.place = defaultdict(lambda: deque(maxlen=self.window))
        self.grasp = defaultdict(lambda: deque(maxlen=self.window))
        self.eplen = defaultdict(lambda: deque(maxlen=self.window))
        self.collision = deque(maxlen=self.window * 4)
        self.occluded = deque(maxlen=self.window * 4)
        self.total_episodes = 0
        self.episodes_at_level = 0

    def _level(self) -> int:
        s = self.state
        return int(s.value) if hasattr(s, "value") else int(s[0])

    def _set_level(self, n: int) -> None:
        if hasattr(self.state, "value"):
            self.state.value = int(n)
        else:
            self.state[0] = int(n)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self.total_episodes += 1
            self.episodes_at_level += 1
            n = int(info.get("num_objects", self.min_objects))
            placed = bool(info.get("place_success", False))
            self.place[n].append(1.0 if placed else 0.0)
            self.grasp[n].append(
                # ever-grasped is the honest signal: is_grasped is False at
                # episode end whenever the cube was placed (grasp released).
                1.0
                if (
                    placed
                    or info.get(
                        "episode_ever_grasped", info.get("is_grasped", False)
                    )
                )
                else 0.0
            )
            self.eplen[n].append(float(info["episode"].get("l", 0)))
            self.collision.append(1.0 if info.get("episode_collision", False) else 0.0)
            self.occluded.append(1.0 if info.get("occluded", False) else 0.0)

        if self.num_timesteps % 1000 == 0 and self.total_episodes > 0:
            self._log_metrics()
            self._maybe_promote()
        return True

    def _log_metrics(self) -> None:
        for n in sorted(self.place):
            if self.place[n]:
                self.logger.record(
                    f"curriculum/place_success_rate/{n}obj",
                    100.0 * float(np.mean(self.place[n])),
                )
                self.logger.record(
                    f"curriculum/grasp_success_rate/{n}obj",
                    100.0 * float(np.mean(self.grasp[n])),
                )
                self.logger.record(
                    f"curriculum/episode_length/{n}obj",
                    float(np.mean(self.eplen[n])),
                )
        if self.collision:
            self.logger.record(
                "curriculum/collision_rate", 100.0 * float(np.mean(self.collision))
            )
        if self.occluded:
            self.logger.record(
                "curriculum/occlusion_rate", 100.0 * float(np.mean(self.occluded))
            )
        self.logger.record("curriculum/num_objects", self._level())
        self.logger.record("curriculum/episodes_at_level", self.episodes_at_level)

    def _maybe_promote(self) -> None:
        n = self._level()
        if n >= self.max_objects:
            return
        bucket = self.place[n]
        ready = (
            self.episodes_at_level >= self.min_episodes_at_level
            and len(bucket) >= self.window
            and float(np.mean(bucket)) >= self.promote_threshold
        )
        if not ready:
            return
        # Save the per-stage best checkpoint, then promote to the next count.
        stage_path = self.model_dir / f"stage_{n}obj_best"
        self.model.save(str(stage_path))
        vec = self.model.get_vec_normalize_env()
        if vec is not None:
            vec.save(str(self.model_dir / f"stage_{n}obj_vecnormalize.pkl"))
        self._set_level(n + 1)
        self.episodes_at_level = 0
        if self.verbose:
            print(
                f"[curriculum] mastered {n} objects "
                f"(place>={self.promote_threshold:.0%}) -> {n + 1}; saved {stage_path}"
            )


def load_pick_place_config(config_path: Optional[str] = None) -> PickPlaceConfig:
    """Load pick-and-place configuration from YAML file."""
    if config_path is None:
        return PickPlaceConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    pick_place_data = raw.get("pick_place", raw)
    return PickPlaceConfig(**pick_place_data)


def save_run_config(
    config: PickPlaceConfig, results_dir: Path, source_path: Optional[str]
) -> None:
    """Save the resolved run config and original YAML when available."""
    with open(results_dir / "resolved_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(mode="json"), fh, sort_keys=False)

    if source_path:
        source = Path(source_path)
        if source.exists():
            shutil.copy2(source, results_dir / "source_config.yaml")


def get_active_task(config: PickPlaceConfig):
    if config.active_task is None:
        return None
    return config.tasks.get(config.active_task)


def setup_directories(config: PickPlaceConfig) -> tuple:
    """Setup training directories."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pick_place_{date_str}"

    results_dir = Path(config.training.log_dir_base) / run_name
    log_dir = results_dir / "logs"
    model_dir = results_dir / "models"

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    return results_dir, log_dir, model_dir


def create_policy_kwargs(config: PickPlaceConfig) -> dict:
    """Create policy kwargs from configuration."""
    net_arch = [int(x) for x in config.training.net_arch.split("-")]
    activation_fn = getattr(nn, config.training.activation_fn, nn.Tanh)

    return {
        "net_arch": {"pi": net_arch, "vf": net_arch},
        "activation_fn": activation_fn,
    }


def create_environment(
    config: PickPlaceConfig,
    n_envs: int,
    eval_env: bool = False,
    global_step_counter: Optional[list] = None,
    object_curriculum_state=None,
):
    """Create training or evaluation environment."""
    task_config = get_active_task(config)
    env_kwargs = {
        "config": config.env,
        "render_mode": None if not eval_env else config.evaluation.render_mode,
        "global_step_counter": global_step_counter,
        "task_config": task_config,
        "object_curriculum_state": object_curriculum_state,
    }

    if n_envs == 1 or eval_env:
        return DummyVecEnv([lambda: Monitor(TentaclePickPlaceEnv(**env_kwargs))])
    else:
        return make_vec_env(
            TentaclePickPlaceEnv,
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
            env_kwargs=env_kwargs,
        )


@app.command()
def train(
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to YAML configuration file"
    ),
    total_timesteps: Optional[int] = typer.Option(None, help="Override total timesteps"),
    num_envs: Optional[int] = typer.Option(None, help="Override number of environments"),
    learning_rate: Optional[float] = typer.Option(None, help="Override learning rate"),
    init_model: Optional[str] = typer.Option(
        None, "--init-model", help="Warm-start from a trained model (.zip); identical obs"
    ),
    verbose: bool = typer.Option(True, help="Enable verbose output"),
) -> None:
    """Train pick-and-place RL model."""
    # Load configuration
    pp_config = load_pick_place_config(config)

    # Apply CLI overrides
    if total_timesteps is not None:
        pp_config.training.total_timesteps = total_timesteps
    if num_envs is not None:
        pp_config.training.num_envs = num_envs
    if learning_rate is not None:
        pp_config.training.learning_rate = learning_rate

    # Setup directories
    results_dir, log_dir, model_dir = setup_directories(pp_config)
    save_run_config(pp_config, results_dir, config)

    # Use a Manager to share the counter cleanly across processes on Windows
    manager = mp.Manager()
    global_step_counter = manager.Value('i', 0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test env to get info
    test_env = TentaclePickPlaceEnv(
        config=pp_config.env,
        global_step_counter=global_step_counter,
        task_config=get_active_task(pp_config),
    )
    steps_per_episode = test_env._max_episode_steps
    time_per_step = test_env.time_per_step
    test_env.close()

    if verbose:
        console.print("[bold green]Pick-and-Place RL Training[/bold green]")
        console.print(f"  Device: [bold cyan]{device.upper()}[/bold cyan]")
        console.print(f"  Total timesteps: {pp_config.training.total_timesteps:,}")
        console.print(f"  Steps/episode: {steps_per_episode}")
        console.print(f"  Time/step: {time_per_step:.3f}s")
        console.print(f"  Learning rate: {pp_config.training.learning_rate}")
        console.print(f"  Num envs: {pp_config.training.num_envs}")
        console.print(f"  Curriculum: {pp_config.env.curriculum_enabled}")
        if pp_config.env.curriculum_enabled:
            console.print(f"    Reach-only: 0 - {pp_config.env.reach_only_steps:,}")
            console.print(f"    Reach+Grasp: {pp_config.env.reach_only_steps:,} - {pp_config.env.reach_grasp_steps:,}")
            console.print(f"    Full: {pp_config.env.reach_grasp_steps:,}+")
        console.print(f"  Results: {results_dir}")

    # Create environments
    train_env = create_environment(
        pp_config, pp_config.training.num_envs,
        global_step_counter=global_step_counter,
    )
    eval_env = create_environment(
        pp_config, 1, eval_env=True,
        global_step_counter=global_step_counter,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    # Create model
    policy_kwargs = create_policy_kwargs(pp_config)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=pp_config.training.learning_rate,
        n_steps=pp_config.training.n_steps,
        batch_size=pp_config.training.batch_size,
        n_epochs=pp_config.training.n_epochs,
        gamma=pp_config.training.gamma,
        gae_lambda=pp_config.training.gae_lambda,
        clip_range=pp_config.training.clip_range,
        ent_coef=pp_config.training.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1 if verbose else 0,
        tensorboard_log=str(log_dir),
        device=device,
    )

    if init_model:
        try:
            model.set_parameters(init_model, exact_match=True)
            if verbose:
                console.print(f"[dim]Warm-started policy from {init_model}[/dim]")
        except Exception as exc:  # noqa: BLE001 - fall back gracefully
            console.print(
                f"[yellow]Warm-start failed ({exc}); training from scratch.[/yellow]"
            )

    # Callbacks
    save_freq = max(1, pp_config.training.save_freq // pp_config.training.num_envs)
    eval_freq = max(1, pp_config.training.eval_freq // pp_config.training.num_envs)

    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(model_dir),
            name_prefix="pick_place_model",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best_model"),
            log_path=str(log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=pp_config.training.n_eval_episodes,
            deterministic=True,
            render=False,
        ),
        PickPlaceMetricsCallback(verbose=1 if verbose else 0),
        SaveVecNormalizeCallback(
            model_dir,
            save_freq=save_freq,
            verbose=1 if verbose else 0,
        ),
    ]

    # Train
    if verbose:
        console.print("[bold]Starting training...[/bold]")

    try:
        model.learn(
            total_timesteps=pp_config.training.total_timesteps,
            callback=callbacks,
            tb_log_name="PPO_PickPlace",
        )

        final_path = model_dir / "final_model"
        model.save(str(final_path))
        vecnorm = model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(str(model_dir / "vecnormalize_final.pkl"))

        if verbose:
            console.print(f"[bold green]Training complete![/bold green]")
            console.print(f"Final model: {final_path}")
            console.print(f"Logs: {log_dir}")

    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise
    finally:
        train_env.close()
        eval_env.close()


@app.command(name="train-multi")
def train_multi(
    config: Optional[str] = typer.Option(
        "rex_tendon/configs/pick_place_multi.yaml",
        "--config",
        help="Multi-object curriculum config",
    ),
    total_timesteps: Optional[int] = typer.Option(None, help="Override total timesteps"),
    num_envs: Optional[int] = typer.Option(None, help="Override number of environments"),
    init_model: Optional[str] = typer.Option(
        None,
        "--init-model",
        help="Warm-start from a trained single-object model (.zip); obs is identical",
    ),
    verbose: bool = typer.Option(True, help="Enable verbose output"),
) -> None:
    """Train a multi-object pick-and-place policy with an adaptive object-count curriculum.

    Ramps spawned objects from min to max (promoting when place-success > 90% at the
    current count), logs per-object-count metrics for the generalization curve, and
    saves a best checkpoint per curriculum stage plus a final multi-object model.
    """
    pp_config = load_pick_place_config(config)
    pp_config.env.object_curriculum_enabled = True
    if total_timesteps is not None:
        pp_config.training.total_timesteps = total_timesteps
    if num_envs is not None:
        pp_config.training.num_envs = num_envs

    results_dir, log_dir, model_dir = setup_directories(pp_config)
    save_run_config(pp_config, results_dir, config)

    manager = mp.Manager()
    global_step_counter = manager.Value("i", 0)
    object_curriculum_state = manager.Value("i", pp_config.env.min_spawned_objects)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        console.print("[bold green]Multi-Object Curriculum Training[/bold green]")
        console.print(
            f"  Objects: {pp_config.env.min_spawned_objects} -> "
            f"{pp_config.env.max_spawned_objects} (adaptive, promote at place>=90%)"
        )
        console.print(
            f"  Device: [cyan]{device.upper()}[/cyan]  "
            f"Envs: {pp_config.training.num_envs}  "
            f"Steps: {pp_config.training.total_timesteps:,}"
        )
        console.print(f"  Results: {results_dir}")

    train_env = create_environment(
        pp_config,
        pp_config.training.num_envs,
        global_step_counter=global_step_counter,
        object_curriculum_state=object_curriculum_state,
    )
    eval_env = create_environment(
        pp_config,
        1,
        eval_env=True,
        global_step_counter=global_step_counter,
        object_curriculum_state=object_curriculum_state,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    policy_kwargs = create_policy_kwargs(pp_config)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=pp_config.training.learning_rate,
        n_steps=pp_config.training.n_steps,
        batch_size=pp_config.training.batch_size,
        n_epochs=pp_config.training.n_epochs,
        gamma=pp_config.training.gamma,
        gae_lambda=pp_config.training.gae_lambda,
        clip_range=pp_config.training.clip_range,
        ent_coef=pp_config.training.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1 if verbose else 0,
        tensorboard_log=str(log_dir),
        device=device,
    )

    if init_model:
        try:
            model.set_parameters(init_model, exact_match=True)
            if verbose:
                console.print(f"[dim]Warm-started policy from {init_model}[/dim]")
        except Exception as exc:  # noqa: BLE001 - fall back gracefully
            console.print(
                f"[yellow]Warm-start failed ({exc}); training from scratch.[/yellow]"
            )

    save_freq = max(1, pp_config.training.save_freq // pp_config.training.num_envs)
    eval_freq = max(1, pp_config.training.eval_freq // pp_config.training.num_envs)
    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(model_dir),
            name_prefix="pick_place_multi",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best_model"),
            log_path=str(log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=pp_config.training.n_eval_episodes,
            deterministic=True,
            render=False,
        ),
        PickPlaceMetricsCallback(verbose=1 if verbose else 0),
        MultiObjectCurriculumCallback(
            object_curriculum_state,
            pp_config.env.min_spawned_objects,
            pp_config.env.max_spawned_objects,
            model_dir,
            verbose=1 if verbose else 0,
        ),
        SaveVecNormalizeCallback(
            model_dir, save_freq=save_freq, verbose=1 if verbose else 0
        ),
    ]

    if verbose:
        console.print("[bold]Starting multi-object curriculum training...[/bold]")
    try:
        model.learn(
            total_timesteps=pp_config.training.total_timesteps,
            callback=callbacks,
            tb_log_name="PPO_PickPlace_Multi",
        )
        final_path = model_dir / "final_model"
        model.save(str(final_path))
        vec = model.get_vec_normalize_env()
        if vec is not None:
            vec.save(str(model_dir / "vecnormalize_final.pkl"))
        if verbose:
            console.print("[bold green]Multi-object training complete![/bold green]")
            console.print(f"Final (multi-object) model: {final_path}")
            console.print(f"Per-stage checkpoints + logs: {model_dir}")
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise
    finally:
        train_env.close()
        eval_env.close()


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to trained model (.zip)"),
    config: Optional[str] = typer.Option(None, "--config", help="Configuration file"),
    num_episodes: Optional[int] = typer.Option(None, help="Number of episodes"),
    render: bool = typer.Option(True, help="Enable rendering"),
    verbose: bool = typer.Option(True, help="Verbose output"),
) -> None:
    """Evaluate a trained pick-and-place RL model."""
    import time

    pp_config = load_pick_place_config(config)

    if num_episodes is not None:
        pp_config.evaluation.num_episodes = num_episodes
    if render:
        pp_config.evaluation.render_mode = "human"

    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        console.print(f"[red]Model not found: {model_path}[/red]")
        raise typer.Exit(1)

    model = PPO.load(str(model_path))

    probe_env = TentaclePickPlaceEnv(
        config=pp_config.env,
        task_config=get_active_task(pp_config),
    )
    model_obs_shape = getattr(model.observation_space, "shape", None)
    if model_obs_shape != probe_env.observation_space.shape and (
        pp_config.env.include_relative_observations
        or pp_config.env.include_object_velocity_in_obs
    ):
        probe_env.close()
        pp_config.env.include_relative_observations = False
        pp_config.env.include_object_velocity_in_obs = False
        probe_env = TentaclePickPlaceEnv(
            config=pp_config.env,
            task_config=get_active_task(pp_config),
        )
        if verbose:
            console.print(
                "[yellow]Using legacy pick-place observation shape for this checkpoint.[/yellow]"
            )
    if model_obs_shape != probe_env.observation_space.shape:
        expected_shape = probe_env.observation_space.shape
        probe_env.close()
        console.print(
            f"[red]Model observation shape {model_obs_shape} does not match env "
            f"shape {expected_shape}.[/red]"
        )
        raise typer.Exit(1)
    probe_env.close()

    # Curriculum disabled for evaluation (always full)
    pp_config.env.curriculum_enabled = False

    env = TentaclePickPlaceEnv(
        config=pp_config.env,
        render_mode=pp_config.evaluation.render_mode,
        task_config=get_active_task(pp_config),
    )

    # Evaluate
    grasp_count = 0
    place_count = 0
    episode_rewards = []

    try:
        for ep in range(pp_config.evaluation.num_episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            done = False

            if verbose:
                console.print(f"\nEpisode {ep + 1}/{pp_config.evaluation.num_episodes}")

            while not done:
                action, _ = model.predict(
                    obs, deterministic=pp_config.evaluation.deterministic_actions
                )
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward

                if pp_config.evaluation.render_mode == "human":
                    env.render()
                    if pp_config.evaluation.render_delay > 0:
                        time.sleep(pp_config.evaluation.render_delay)

            episode_rewards.append(ep_reward)
            if info.get("is_grasped", False) or info.get("place_success", False):
                grasp_count += 1
            if info.get("place_success", False):
                place_count += 1

            if verbose:
                console.print(f"  Reward: {ep_reward:.3f}")
                console.print(f"  Grasp: {'Yes' if info.get('is_grasped') or info.get('place_success') else 'No'}")
                console.print(f"  Place: {'Yes' if info.get('place_success') else 'No'}")

    except KeyboardInterrupt:
        console.print("\nEvaluation interrupted.")
    finally:
        env.close()

    # Results
    if episode_rewards:
        n = len(episode_rewards)
        console.print(f"\n{'=' * 50}")
        console.print("[bold]EVALUATION RESULTS[/bold]")
        console.print(f"{'=' * 50}")
        console.print(f"Episodes: {n}")
        console.print(f"Mean reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        console.print(f"Grasp rate: {grasp_count / n * 100:.1f}%")
        console.print(f"Place rate: {place_count / n * 100:.1f}%")


@app.command(name="eval-curve")
def eval_curve(
    model_path: str = typer.Argument(..., help="Trained model .zip (e.g. final_model.zip)"),
    config: Optional[str] = typer.Option(
        "rex_tendon/configs/pick_place_multi.yaml", "--config", help="Env/task config"
    ),
    counts: str = typer.Option("2,3,4,5", "--counts", help="Comma-separated object counts"),
    episodes: int = typer.Option(100, "--episodes", help="Held-out episodes per count"),
    seed_base: int = typer.Option(
        1_000_000, "--seed-base", help="Held-out seed offset (disjoint from training)"
    ),
    output: Optional[str] = typer.Option(None, "--output", help="CSV output path"),
) -> None:
    """Held-out generalization eval: final policy x fixed object count -> CSV.

    Runs the deterministic policy at each fixed object count on held-out seeds and
    writes grasp/place/collision/occlusion rates + mean episode length per count.
    This is the clean curve for thesis/paper figures (vs the noisier on-policy
    curriculum tags logged during training).
    """
    import csv

    mp_path = Path(model_path)
    if not mp_path.exists():
        console.print(f"[red]Model not found: {mp_path}[/red]")
        raise typer.Exit(1)
    model = PPO.load(str(mp_path))
    pp_config = load_pick_place_config(config)
    count_list = [int(c) for c in str(counts).split(",")]

    def make_env(n: int) -> TentaclePickPlaceEnv:
        env_cfg = pp_config.env.model_copy(deep=True)
        env_cfg.object_curriculum_enabled = False
        env_cfg.curriculum_enabled = False  # always evaluate in the FULL phase
        env_cfg.num_spawned_objects = n
        return TentaclePickPlaceEnv(
            config=env_cfg, task_config=get_active_task(pp_config)
        )

    # Load the policy's observation normalisation (count-independent).
    def normalize(obs):
        return obs

    stats_env = None
    for d in (mp_path.parent, mp_path.parent.parent):
        if (d / "vecnormalize.pkl").exists():
            stats_env = VecNormalize.load(
                str(d / "vecnormalize.pkl"),
                DummyVecEnv([lambda: make_env(count_list[0])]),
            )
            stats_env.training = False
            normalize = stats_env.normalize_obs
            break
    if stats_env is None:
        console.print("[yellow]No vecnormalize.pkl found; using raw observations.[/yellow]")

    console.print(
        f"[bold]Held-out generalization eval[/bold] ({episodes} eps/count, "
        f"seeds {seed_base}+):"
    )
    rows = []
    for n in count_list:
        env = make_env(n)
        place = grasp = coll = occ = 0
        lengths = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed_base + ep)
            done, steps = False, 0
            ep_place = ep_grasp = ep_coll = ep_occ = False
            while not done:
                action, _ = model.predict(normalize(obs), deterministic=True)
                obs, _r, term, trunc, info = env.step(action)
                done = bool(term or trunc)
                steps += 1
                ep_place = ep_place or bool(info.get("place_success", False))
                ep_grasp = ep_grasp or bool(info.get("is_grasped", False))
                ep_coll = ep_coll or bool(info.get("episode_collision", False))
                ep_occ = bool(info.get("occluded", False))
            place += int(ep_place)
            grasp += int(ep_place or ep_grasp)
            coll += int(ep_coll)
            occ += int(ep_occ)
            lengths.append(steps)
        env.close()
        row = {
            "num_objects": n,
            "episodes": episodes,
            "grasp_success_rate": round(100.0 * grasp / episodes, 2),
            "place_success_rate": round(100.0 * place / episodes, 2),
            "collision_rate": round(100.0 * coll / episodes, 2),
            "occlusion_rate": round(100.0 * occ / episodes, 2),
            "mean_episode_length": round(float(np.mean(lengths)), 1),
        }
        rows.append(row)
        console.print(
            f"  {n} obj: place={row['place_success_rate']}%  "
            f"grasp={row['grasp_success_rate']}%  "
            f"collision={row['collision_rate']}%  "
            f"occ={row['occlusion_rate']}%  eplen={row['mean_episode_length']}"
        )

    if stats_env is not None:
        stats_env.close()
    out = Path(output) if output else mp_path.parent / "generalization_curve.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[bold green]Saved generalization curve -> {out}[/bold green]")


if __name__ == "__main__":
    app()

