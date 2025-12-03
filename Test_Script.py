import gymnasium as gym
import register_env
from stable_baselines3 import SAC

env = gym.make("DM-v0")
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
