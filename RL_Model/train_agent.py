import gymnasium as gym
import os
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from new_inteli_env import UdupiDeliveryEnv

# Instantiate environment
env = UdupiDeliveryEnv()

# Optional: check if the env follows Gym interface
check_env(env, warn=True)

# Create DQN agent
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.0005,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    tau=0.01,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device='auto'
)

# Train agent
model.learn(total_timesteps=100_000)

# Save model
model.save("udupi_dqn_model")
print("✅ Model trained and saved!")
