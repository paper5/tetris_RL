import cv2
import gymnasium as gym
from collections import defaultdict
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.dqn.policies import DQNPolicy
from tetris_gymnasium.envs.tetris import Tetris
from stable_baselines3 import DQN


print("starting....")
# Load your trained model
model = DQN.load("best_model/best_model.zip")

# Create environment with video recording
env = gym.make(
    'tetris_gymnasium/Tetris',
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


#may 8: its not very good
#makes 50 episodes
#check if it can spin
#todo: put all one video and speed up