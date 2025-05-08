import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo


# Load your trained model
model = PPO.load("tetris_ppo")

# Create environment with video recording
env = gym.make(
    'tetris_gymnasium/Tetris-v0',
    obs_type='grayscale',  # or 'rgb'
    render_mode='rgb_array'  # Required for video recording
)

# Wrap the env to record videos (saved in ./videos/)
env = RecordVideo(
    env,
    "videos",
    episode_trigger=lambda x: True,  # Record every episode
    name_prefix="tetris_agent"  # Custom filename
)

# Run the agent
obs, _ = env.reset()
for _ in range(5000):  # Run for 5000 steps (adjust as needed)
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
