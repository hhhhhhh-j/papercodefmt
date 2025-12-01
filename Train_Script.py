import gymnasium as gym
import register_env
from stable_baselines3 import SAC
from custom_cnn import CustomCNN

print("Start Training")
env = gym.make("DM-v0")

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256)
)

model = SAC(
    "MultiInputPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    buffer_size=20000,
    batch_size=64
)

# model = SAC.load("sac_decision_making", env=env)
model.learn(total_timesteps=8000, log_interval=4)
model.save("sac_decision_making")

env.close()
