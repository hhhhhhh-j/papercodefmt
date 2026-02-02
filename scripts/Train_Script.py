import os, sys
import multiprocessing as mp
import gymnasium as gym
from loguru import logger

def main():
    mp.set_start_method("forkserver", force=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)
    sys.path.append(ROOT_DIR)
    sys.path.append(os.path.join(ROOT_DIR, "utils"))
    MODEL_PATH = os.path.join(ROOT_DIR, "wandb_models", "model.zip")

    print("PYTHONPATH updated:", ROOT_DIR)

    import utils.register_env as register_env

    import wandb
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from interactions.custom_encoder import CustomEncoder
    from wandb.integration.sb3 import WandbCallback
    from utils.wandb_callback import WandbCustomCallback

    logger.info("Start Training")

    # env = make_vec_env(
    #     "DM-v1",
    #     n_envs=16,
    #     vec_env_cls=SubprocVecEnv,
    #     vec_env_kwargs=dict(start_method="spawn"),
    #     monitor_dir="./monitor"
    # )
    def make_env(rank: int):
        def _init():
            import gymnasium as gym          # ✅ 关键：子进程里也 import
            import utils.register_env        # ✅ 关键：子进程里注册
            from stable_baselines3.common.monitor import Monitor
            env = gym.make("DM-v1")
            env = Monitor(env)
            return env
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(16)])

    wandb.init(
        sync_tensorboard=True,
        dir="./",
        project="offroad-sac-ds",
        name="SAC_DS_run1",
        config={
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 512,
            "gamma": 0.99,
            "tau": 0.005,
            "features_dim": 256,
            "total_timesteps": 50_0000,  # 时间紧张先别 1e9
        }
    )

    policy_kwargs = dict(
        features_extractor_class=CustomEncoder,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256])
    )

    model = SAC(
        "MultiInputPolicy",
        env,
        device="cuda",
        learning_rate=wandb.config.learning_rate,
        buffer_size=wandb.config.buffer_size,
        batch_size=wandb.config.batch_size,
        tau=wandb.config.tau,
        gamma=wandb.config.gamma,
        train_freq=(1, "step"),
        gradient_steps=4,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=policy_kwargs,
        verbose=2,
        tensorboard_log="./tb_logs/",
        learning_starts=5000,
    )
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading weights from: {MODEL_PATH}")
        loaded = SAC.load(MODEL_PATH, env=env, device="cuda")
        model.set_parameters(loaded.get_parameters(), exact_match=True)

    wandb_cb = WandbCallback(
        model_save_path="./wandb_models",
        model_save_freq=5000,
        verbose=2
    )

    custom_cb = WandbCustomCallback(log_freq=2000)

    model.learn(
        total_timesteps=wandb.config.total_timesteps,
        log_interval=4,
        callback=[wandb_cb, custom_cb]
    )

    model.save("./sac_decision_making_v2")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    # 如果未来要 freeze 成可执行文件，需要 freeze_support；普通训练可不加
    # mp.freeze_support()
    main()
