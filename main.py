import cv2
import gymnasium as gym
from collections import defaultdict
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv
from tetris_gymnasium.envs.tetris import Tetris

env = gym.make('tetris_gymnasium/Tetris')
vec_env = make_vec_env(lambda: gym.make('tetris_gymnasium/Tetris'), n_envs=4) #for parallel enviorments

model = PPO("MultiInputPolicy", vec_env,
    verbose=1,
    n_steps=2048,  # Longer horizons for planning
    batch_size=64,
    n_epochs=10,
    tensorboard_log="./ppo_tetris_logs/")
#more steps for advance

eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    render=False
)
print("Ready!")
model.learn(total_timesteps=1000000, callback=eval_callback)

# Save the model
model.save("tetris_ppo")


#