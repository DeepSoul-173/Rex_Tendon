"""RL training module with PPO and custom environment."""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import shutil

import typer
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from rich.console import Console

from .environment import TentacleTargetFollowingEnv
from .evaluation import evaluate_rl_model
from ...configs.loaders import get_rl_training_config
from ...configs.rl_training import RLTrainingConfig

logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer(help="RL training utilities")


def setup_directories(config: RLTrainingConfig) -> tuple[Path, Path, Path]:
    """Setup training directories with timestamp."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_tentacle_{date_str}"

    results_dir = Path(config.rl_training_params.log_dir_base) / run_name
    log_dir = results_dir / "logs"
    model_dir = results_dir / "models"

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    return results_dir, log_dir, model_dir


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


def create_policy_kwargs(config: RLTrainingConfig) -> dict:
    """Create policy kwargs from configuration."""
    net_arch = [int(x) for x in config.rl_training_params.net_arch.split("-")]
    activation_fn = getattr(nn, config.rl_training_params.activation_fn, nn.Tanh)

    return {
        "net_arch": {
            "pi": net_arch,
            "vf": net_arch,
        },
        "activation_fn": activation_fn,
    }


def create_environment(config: RLTrainingConfig, n_envs: int, eval_env: bool = False):
    """Create training or evaluation environment."""
    env_kwargs = {
        "config": config.rl_env,
        "render_mode": None if not eval_env else config.rl_evaluation.render_mode,
    }

    if (
        eval_env
        and config.rl_evaluation.render_mode == "human"
        and sys.platform == "darwin"
    ):
        logger.warning("\n  WARNING: macOS detected with render_mode='human'")
        logger.warning("   MuJoCo viewer requires 'mjpython' on macOS.")
        logger.warning("   Consider setting render_mode=None in config for training,")
        logger.warning(
            "   or run evaluation separately with: mjpython rex-tendon train rl"
        )

    if n_envs == 1 or eval_env:
        return DummyVecEnv([lambda: Monitor(TentacleTargetFollowingEnv(**env_kwargs))])
    else:
        return make_vec_env(
            TentacleTargetFollowingEnv,
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv,
            env_kwargs=env_kwargs,
        )


def create_callbacks(
    config: RLTrainingConfig, log_dir: Path, model_dir: Path, eval_env
):
    """Create training callbacks."""
    callbacks = []

    # save_freq and eval_freq are total steps across all envs
    # Need to divide by num_envs to get the per-environment frequency
    save_freq_per_env = max(
        1, config.rl_training_params.save_freq // config.rl_training_params.num_envs
    )
    eval_freq_per_env = max(
        1, config.rl_training_params.eval_freq // config.rl_training_params.num_envs
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=str(model_dir),
        name_prefix="tentacle_model",
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(log_dir),
        eval_freq=eval_freq_per_env,
        n_eval_episodes=config.rl_training_params.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    callbacks.append(
        SaveVecNormalizeCallback(
            model_dir,
            save_freq=save_freq_per_env,
            verbose=0,
        )
    )

    return callbacks


def train_rl_model(
    config: Optional[str] = typer.Option(None, help="Path to YAML configuration file"),
    total_timesteps: Optional[int] = typer.Option(
        None, help="Override total timesteps"
    ),
    num_envs: Optional[int] = typer.Option(
        None, help="Override number of environments"
    ),
    learning_rate: Optional[float] = typer.Option(None, help="Override learning rate"),
    verbose: bool = typer.Option(True, help="Enable verbose output"),
) -> None:
    """Train RL model using centralized configuration.

    Args:
        config: Optional YAML config file to override defaults
        total_timesteps: Override total training timesteps
        num_envs: Override number of parallel environments
        learning_rate: Override learning rate
        verbose: Enable detailed logging
    """
    # Load configuration
    source_config_path = config
    config = get_rl_training_config(config)

    # Apply CLI overrides
    if total_timesteps is not None:
        config.rl_training_params.total_timesteps = total_timesteps
    if num_envs is not None:
        config.rl_training_params.num_envs = num_envs
    if learning_rate is not None:
        config.rl_training_params.learning_rate = learning_rate

    # Setup directories
    results_dir, log_dir, model_dir = setup_directories(config)

    # Save configuration used for this run
    resolved_config_path = results_dir / "resolved_config.yaml"
    with open(resolved_config_path, "w", encoding="utf-8") as fh:
        import yaml

        yaml.safe_dump(config.model_dump(mode="json"), fh, sort_keys=False)
    if source_config_path:
        try:
            shutil.copy2(source_config_path, results_dir / "source_config.yaml")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save source config: {e}[/yellow]")

    # Create test environment to get derived parameters
    test_env = TentacleTargetFollowingEnv(config=config.rl_env)
    steps_per_episode = test_env._max_episode_steps
    time_per_step = test_env.time_per_step
    test_env.close()

    if verbose:
        console.print("Training configuration:")
        console.print(
            f"  - Total timesteps: {config.rl_training_params.total_timesteps:,}"
        )
        console.print(f"  - Steps per episode: {steps_per_episode}")
        console.print(f"  - Time per step: {time_per_step:.3f}s")
        console.print(f"  - Learning rate: {config.rl_training_params.learning_rate}")
        console.print(
            f"  - Number of environments: {config.rl_training_params.num_envs}"
        )
        console.print(
            f"  - Save frequency: {config.rl_training_params.save_freq} total steps ({config.rl_training_params.save_freq // config.rl_training_params.num_envs} per env)"
        )
        console.print(
            f"  - Eval frequency: {config.rl_training_params.eval_freq} total steps ({config.rl_training_params.eval_freq // config.rl_training_params.num_envs} per env)"
        )
        console.print(f"  - Dynamics randomization: {config.rl_env.randomize_dynamics}")
        console.print(f"  - Observation noise: {config.rl_env.add_observation_noise}")
        console.print(f"  - Results directory: {results_dir}")

    # Create environments
    train_env = create_environment(config, config.rl_training_params.num_envs)
    eval_env = create_environment(config, 1, eval_env=True)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False

    # Setup policy
    policy_kwargs = create_policy_kwargs(config)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.rl_training_params.learning_rate,
        n_steps=config.rl_training_params.n_steps,
        batch_size=config.rl_training_params.batch_size,
        n_epochs=config.rl_training_params.n_epochs,
        gamma=config.rl_training_params.gamma,
        gae_lambda=config.rl_training_params.gae_lambda,
        clip_range=config.rl_training_params.clip_range,
        ent_coef=config.rl_training_params.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1 if verbose else 0,
        tensorboard_log=str(log_dir),
    )

    # Setup callbacks
    callbacks = create_callbacks(config, log_dir, model_dir, eval_env)

    if verbose:
        console.print("Starting training...")

    # Train the model
    try:
        model.learn(
            total_timesteps=config.rl_training_params.total_timesteps,
            callback=callbacks,
            tb_log_name="PPO",
        )

        # Save final model
        final_model_path = model_dir / "final_model"
        model.save(str(final_model_path))
        vecnorm = model.get_vec_normalize_env()
        if vecnorm is not None:
            vecnorm.save(str(model_dir / "vecnormalize_final.pkl"))

        if verbose:
            console.print(f"Training completed successfully!")
            console.print(f"Final model saved to: {final_model_path}")
            console.print(f"Logs saved to: {log_dir}")

    except Exception as e:
        console.print(f"[red]Training failed with error: {e}[/red]")
        raise
    finally:
        # Clean up environments
        train_env.close()
        eval_env.close()


@app.command()
def train(
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    total_timesteps: Optional[int] = typer.Option(
        None, help="Override total timesteps"
    ),
    num_envs: Optional[int] = typer.Option(
        None, help="Override number of environments"
    ),
    learning_rate: Optional[float] = typer.Option(None, help="Override learning rate"),
    verbose: bool = typer.Option(True, help="Enable verbose output"),
) -> None:
    """Train RL model using centralized configuration."""
    train_rl_model(config, total_timesteps, num_envs, learning_rate, verbose)


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to the trained model (.zip file)"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    num_episodes: Optional[int] = typer.Option(
        None, help="Override number of evaluation episodes"
    ),
    render: bool = typer.Option(True, help="Enable rendering during evaluation"),
    deterministic: bool = typer.Option(
        True, help="Use deterministic actions"
    ),
    render_delay: Optional[float] = typer.Option(
        None, help="Delay between frames (seconds)"
    ),
    save_results: bool = typer.Option(False, help="Save evaluation results to file"),
    verbose: bool = typer.Option(True, help="Enable verbose output"),
) -> None:
    """Evaluate a trained RL model."""
    evaluate_rl_model(
        model_path,
        config,
        num_episodes,
        render,
        deterministic,
        render_delay,
        save_results,
        verbose,
    )


if __name__ == "__main__":
    app()

