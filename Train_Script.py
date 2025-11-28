import gymnasium as gym
import register_env
from stable_baselines3 import SAC

print("1. Start")
env = gym.make("DM-v0")

# model = SAC("MultiInputPolicy", env, verbose=1)
model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    buffer_size=20000,    # 从 1000000 → 20000
    batch_size=64
)
# print(model.policy)
# print("Actor network:", model.policy.actor)
# print("Critic network:", model.policy.critic)
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_decision_making")

del model # remove to demonstrate saving and loading

model = SAC.load("sac_decision_making", env=env)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

