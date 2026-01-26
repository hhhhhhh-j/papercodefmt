import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))

print("PYTHONPATH updated:", ROOT_DIR)

import gymnasium as gym
import utils.register_env as register_env
import wandb
from stable_baselines3 import SAC
from utils.custom_encoder import CustomEncoder
from stable_baselines3.common.callbacks import BaseCallback
from loguru import logger
from wandb.integration.sb3 import WandbCallback
from utils.wandb_callback import WandbCustomCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# logging
logger.add("logging/train.log", rotation="50 MB", retention="7 days", compression="zip")
# logger.remove() # 去掉默认 handler，自定义
# logger.add(sys.stdout, level="INFO")
# logger.add("logging/train_debug.log", level="DEBUG", rotation="50 MB", retention="14 days")


print("Start Training")
env = make_vec_env("DM-v1", n_envs=1, monitor_dir="./monitor")

# 初始化 wandb
wandb.init(
    sync_tensorboard=True,
    dir="./",
    project="offroad-sac-ds",       # 你的项目名称
    name="SAC_DS_run1",             # 当前实验名
    config={
        "learning_rate": 3e-4,
        "buffer_size": 10000,       # 1000000
        "batch_size": 32,           # 256
        "gamma": 0.99,
        "tau": 0.005,
        "net_arch": [256, 256],
        "features_dim": 256, 
        "total_timesteps": 1000000,
    }
)

policy_kwargs = dict(
    features_extractor_class=CustomEncoder,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=dict(
        pi=[256, 256],         # actor
        qf=[256, 256]          # critic
    )
)

# 创建或加载 SAC 模型
print("CWD =", os.getcwd())
print("Expect ckpt =", os.path.abspath("sac_decision_making.zip"))
print("Exists? =", os.path.exists("sac_decision_making.zip"))
if os.path.exists("sac_decision_making.zip"):
    print("Loading existing model...")
    model = SAC.load("sac_decision_making", env=env)
else:
    print("Creating new model...")
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=wandb.config.learning_rate,
        buffer_size=wandb.config.buffer_size,
        batch_size=wandb.config.batch_size,
        tau=wandb.config.tau,
        gamma=wandb.config.gamma,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        verbose=2, 
        tensorboard_log="./tb_logs/", 
    )

# 开始训练（接入 WandB）
wandb_cb = WandbCallback(
    model_save_path="./wandb_models",
    model_save_freq=5000,
    verbose=2
)

custom_cb = WandbCustomCallback(
    log_freq=200,
)

model.learn(
    total_timesteps=wandb.config.total_timesteps,
    log_interval=4,
    callback=[wandb_cb, custom_cb]
)

model.save("/home/fmt/catkin_ws/src/fmt_/sb3_SAC/sac_decision_making")
env.close()
wandb.finish()

if __name__ == "__main__":
    '''
    目前仍存在的问题
    1.如何针对超参数进行优化
    2.如何自动的生成随机地图进行训练
    3. ...
    '''
    pass