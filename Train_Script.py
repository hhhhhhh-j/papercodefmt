import gymnasium as gym
import register_env
from stable_baselines3 import SAC
from custom_cnn import CustomCNN
from stable_baselines3.common.callbacks import BaseCallback

print("Start Training")
env = gym.make("DM-v0")

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(
        pi=[256, 256],         # actor
        qf=[256, 256]          # critic
    )
)

model = SAC(
    policy="MultiInputPolicy",
    env=env,
    tensorboard_log="./logs_sac",
    learning_rate=3e-4,        # ★ 默认3e-4，适合大多数 SAC 应用
    buffer_size=200000,        # ★ 越野 RL 必须要大 buffer（你的 20000 绝对不够）
    batch_size=256,            # ★ SAC 使用较大 batch 更稳定
    tau=0.005,                 # target smoothing
    gamma=0.99,                # discount factor
    train_freq=(1, "step"),    # 每步更新
    gradient_steps=1,          # 每次更新1步（可提高）
    ent_coef="auto",           # ★ 自动调节 entropy 更稳定
    target_entropy="auto",     # 自动
    use_sde=False,
    policy_kwargs=policy_kwargs,
    verbose=1
)

model = SAC.load("sac_decision_making", env=env, tensorboard_log="./logs_sac")
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_decision_making")

env.close()

if __name__ == "__main__":
    '''
    目前仍存在的问题
    1.如何针对超参数进行优化
    2.如何自动的生成随机地图进行训练
    3. ...
    '''
    pass