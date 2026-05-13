"""Refactored RL evaluation script using centralized configuration."""

import time
from pathlib import Path
from typing import Optional

import typer
import numpy as np
from stable_baselines3 import PPO
from rich.console import Console

from .environment import TentacleTargetFollowingEnv
from ...configs.loaders import get_rl_training_config

console = Console()


def evaluate_rl_model(
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
    """Evaluate a trained RL model using centralized configuration.

    Args:
        model_path: Path to the trained model file
        config: Optional YAML config file to override defaults
        num_episodes: Override number of episodes to evaluate
        render: Enable visual rendering
        deterministic: Use deterministic policy actions
        render_delay: Delay between rendered frames
        save_results: Save detailed results to file
        verbose: Enable detailed logging
    """
    # Load configuration
    config = get_rl_training_config(config)

    # Apply CLI overrides
    if num_episodes is not None:
        config.rl_evaluation.num_episodes = num_episodes
    if deterministic is not None:
        config.rl_evaluation.deterministic_actions = deterministic
    if render_delay is not None:
        config.rl_evaluation.render_delay = render_delay
    if render:
        config.rl_evaluation.render_mode = "human"
    else:
        config.rl_evaluation.render_mode = "none"

    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        console.print(f"[red]Error: Model file not found: {model_path}[/red]")
        raise typer.Exit(1)

    if verbose:
        console.print(f"Loading model from: {model_path}")

    try:
        model = PPO.load(str(model_path))
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(1)

    # Create evaluation environment
    env = TentacleTargetFollowingEnv(
        config=config.rl_env,
        render_mode=config.rl_evaluation.render_mode,
    )

    if verbose:
        console.print("Evaluation configuration:")
        console.print(f"  - Number of episodes: {config.rl_evaluation.num_episodes}")
        console.print(
            f"  - Deterministic actions: {config.rl_evaluation.deterministic_actions}"
        )
        console.print(f"  - Render mode: {config.rl_evaluation.render_mode}")
        console.print(f"  - Render delay: {config.rl_evaluation.render_delay}s")

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    final_distances = []
    success_episodes = 0
    success_threshold = 0.05  # Define success as being within 5cm of target

    try:
        for episode in range(config.rl_evaluation.num_episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            if verbose:
                console.print(
                    f"\nEpisode {episode + 1}/{config.rl_evaluation.num_episodes}"
                )

            while not done:
                # Get action from model
                action, _states = model.predict(
                    obs, deterministic=config.rl_evaluation.deterministic_actions
                )

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                # Render if enabled
                if config.rl_evaluation.render_mode == "human":
                    env.render()
                    if config.rl_evaluation.render_delay > 0:
                        time.sleep(config.rl_evaluation.render_delay)

            # Record metrics
            final_distance = info.get("distance", info.get("distance_to_target", float("inf")))
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            final_distances.append(final_distance)

            if final_distance <= success_threshold:
                success_episodes += 1

            if verbose:
                console.print(f"  - Reward: {episode_reward:.3f}")
                console.print(f"  - Length: {episode_length} steps")

    except KeyboardInterrupt:
        console.print("\nEvaluation interrupted by user.")
    finally:
        env.close()

    # Calculate statistics
    if episode_rewards:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        mean_distance = np.mean(final_distances)
        success_rate = (success_episodes / len(episode_rewards)) * 100

        # Display results
        console.print(f"\n{'='*50}")
        console.print("EVALUATION RESULTS")
        console.print(f"{'='*50}")
        console.print(f"Episodes completed: {len(episode_rewards)}")
        console.print(f"Mean episode reward: {mean_reward:.3f} ± {std_reward:.3f}")
        console.print(f"Mean episode length: {mean_length:.1f} steps")
        console.print(f"Mean final distance: {mean_distance:.4f}m")
        console.print(f"Success rate: {success_rate:.1f}% (< {success_threshold}m)")
        console.print(f"Best episode reward: {max(episode_rewards):.3f}")
        console.print(f"Worst episode reward: {min(episode_rewards):.3f}")
        console.print(f"Best final distance: {min(final_distances):.4f}m")

        # Save results if requested
        if save_results:
            results_file = (
                model_path.parent / f"evaluation_results_{model_path.stem}.txt"
            )
            with open(results_file, "w") as f:
                f.write(f"Evaluation Results for {model_path}\n")
                f.write(f"{'='*50}\n")
                f.write(f"Episodes completed: {len(episode_rewards)}\n")
                f.write(f"Mean episode reward: {mean_reward:.3f} ± {std_reward:.3f}\n")
                f.write(f"Mean episode length: {mean_length:.1f} steps\n")
                f.write(f"Mean final distance: {mean_distance:.4f}m\n")
                f.write(f"Success rate: {success_rate:.1f}% (< {success_threshold}m)\n")
                f.write(f"Best episode reward: {max(episode_rewards):.3f}\n")
                f.write(f"Worst episode reward: {min(episode_rewards):.3f}\n")
                f.write(f"Best final distance: {min(final_distances):.4f}m\n")
                f.write(f"\nDetailed Results:\n")
                for i, (reward, length, distance) in enumerate(
                    zip(episode_rewards, episode_lengths, final_distances)
                ):
                    f.write(
                        f"Episode {i+1}: Reward={reward:.3f}, Length={length}, Distance={distance:.4f}m\n"
                    )

            console.print(f"\nResults saved to: {results_file}")
    else:
        console.print("No episodes completed.")

