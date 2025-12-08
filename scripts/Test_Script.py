import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(ROOT_DIR)

print("PYTHONPATH updated:", ROOT_DIR)

import gymnasium as gym
import utils.register_env as register_env
from stable_baselines3 import SAC

env = gym.make("DM-v1")
model = SAC.load("sac_decision_making", env=env)

obs, info = env.reset()

for _ in range(1000):  
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    print("pose:", obs["pose"], "reward:", reward)

    if terminated or truncated:
        print("Episode finished")
        break

env.close()
