"""End-to-end test: short RL training run to verify the full pipeline."""
import sys
sys.path.insert(0, ".")
import numpy as np

print("=" * 60)
print("  END-TO-END PICK-AND-PLACE TEST")
print("=" * 60)

# ===== Test 1: MuJoCo Scene =====
print("\n[1/5] Testing MuJoCo scene loading...")
import mujoco
m = mujoco.MjModel.from_xml_path("rex_assets/rex_simulation/pick_and_place_scene.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tip_center")
zone_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "place_zone")
cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_cube")
cyl_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_cylinder")
sph_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_sphere")
grasp_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_cube")

print(f"  Bodies: {m.nbody}, Joints: {m.njnt}, Actuators: {m.nu}")
print(f"  Tendons: {m.ntendon}, Eq constraints: {m.neq}")
print(f"  Tip site: {tip_id} -> pos {d.site_xpos[tip_id]}")
print(f"  Place zone: {zone_id} -> pos {d.site_xpos[zone_id]}")
print(f"  Cube body: {cube_id} -> pos {d.xpos[cube_id]}")
# Equality constraints removed for programmatic grasping (grasp_id will be -1)
assert tip_id >= 0 and zone_id >= 0 and cube_id >= 0
print("  PASS")

# ===== Test 2: Environment =====
print("\n[2/5] Testing Gymnasium environment...")
from rex_tendon.training.rl.pick_place_env import TentaclePickPlaceEnv
from rex_tendon.configs.pick_place_config import PickPlaceEnvConfig

config = PickPlaceEnvConfig(
    curriculum_enabled=False,  # Test full mode directly
    randomize_dynamics=False,
    add_observation_noise=False,
    simulation_length_seconds=3.0,
)
env = TentaclePickPlaceEnv(config=config)
obs, info = env.reset()

assert obs.shape == (60,), f"Expected (60,), got {obs.shape}"
assert env.action_space.shape == (2,)
print(f"  Obs: {obs.shape}, Action: {env.action_space.shape}")

# Step test
rewards = []
for i in range(20):
    action = np.random.uniform(-0.5, 0.5, size=2).astype(np.float32)
    obs, rew, term, trunc, info = env.step(action)
    rewards.append(rew)
    if term or trunc:
        obs, info = env.reset()

print(f"  20 steps: mean_reward={np.mean(rewards):.4f}")
print(f"  tip_to_obj={info.get('tip_to_object_distance', 'N/A'):.4f}")
env.close()
print("  PASS")

print("\n[3/5] Testing config loading...")
from rex_tendon.configs.pick_place_config import PickPlaceConfig
config = PickPlaceConfig()
# Updated threshold to 0.06 for soft programmatic grasp
assert config.env.grasp_distance_threshold == 0.06
assert config.training.total_timesteps == 10_000_000
assert config.env.curriculum_enabled == True
print(f"  Default config: {config.training.total_timesteps} timesteps")
print(f"  Curriculum: reach_only={config.env.reach_only_steps}, reach_grasp={config.env.reach_grasp_steps}")
print("  PASS")

# ===== Test 4: Short PPO training (500 steps) =====
print("\n[4/5] Testing short PPO training (500 steps)...")
from stable_baselines3 import PPO

train_config = PickPlaceEnvConfig(
    curriculum_enabled=False,
    randomize_dynamics=False,
    add_observation_noise=False,
    simulation_length_seconds=2.0,
)
train_env = TentaclePickPlaceEnv(config=train_config)

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=64,
    batch_size=32,
    n_epochs=2,
    verbose=0,
)
model.learn(total_timesteps=500)
print("  Training completed (500 steps)")

# Test inference
obs, _ = train_env.reset()
action, _ = model.predict(obs, deterministic=True)
assert action.shape == (2,), f"Expected action (2,), got {action.shape}"
print(f"  Inference OK: action={action}")

train_env.close()
print("  PASS")

# ===== Test 5: Curriculum phases =====
print("\n[5/5] Testing curriculum learning phases...")
from rex_tendon.training.rl.pick_place_env import PickPlacePhase

counter = [0]
curriculum_config = PickPlaceEnvConfig(
    curriculum_enabled=True,
    reach_only_steps=100,
    reach_grasp_steps=200,
    simulation_length_seconds=2.0,
    randomize_dynamics=False,
    add_observation_noise=False,
)
cenv = TentaclePickPlaceEnv(config=curriculum_config, global_step_counter=counter)

# Phase REACH
counter[0] = 50
obs, _ = cenv.reset()
assert cenv._get_curriculum_phase() == PickPlacePhase.REACH
print(f"  Step {counter[0]}: Phase REACH")

# Phase REACH_GRASP
counter[0] = 150
obs, _ = cenv.reset()
assert cenv._get_curriculum_phase() == PickPlacePhase.REACH_GRASP
print(f"  Step {counter[0]}: Phase REACH_GRASP")

# Phase FULL
counter[0] = 250
obs, _ = cenv.reset()
assert cenv._get_curriculum_phase() == PickPlacePhase.FULL
print(f"  Step {counter[0]}: Phase FULL")

cenv.close()
print("  PASS")

print("\n" + "=" * 60)
print("  ALL TESTS PASSED!")
print("=" * 60)


