from stable_baselines3 import DQN
from new_inteli_env import UdupiDeliveryEnv
import time

env = UdupiDeliveryEnv()
model = DQN.load("udupi_dqn_model")

obs, _ = env.reset()  # ✅ FIXED HERE
done = False
total_reward = 0

while not done:
    env.render()
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)  # ✅ Gymnasium style
    done = terminated or truncated
    total_reward += reward
    time.sleep(1)

print(f"Total Reward: {total_reward}")

env.close()