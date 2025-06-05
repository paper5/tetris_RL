import cv2
import gymnasium as gym
from collections import defaultdict
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv
from tetris_gymnasium.envs.tetris import Tetris

env = gym.make('tetris_gymnasium/Tetris')
# DQN does not support parallel environments

model = DQN("MultiInputPolicy", env,
    verbose=1,
    learning_rate=1e-4,  # Adjusted learning rate for DQN
    buffer_size=100000,  # Replay buffer size
    learning_starts=1000,  # Steps before training starts
    batch_size=128,  # Batch size for training
    target_update_interval=1000,  # Target network update frequency
    train_freq=(4, "step"),  # Frequency of training
    tensorboard_log="./dqn_tetris_logs/")

eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10000,  # Reduced frequency for DQN
    deterministic=True,
    render=False #don't render the training
)
print("Ready!")
model.learn(total_timesteps=16000000, callback=eval_callback)

# Save the model
model.save("tetris_dqn")


#
