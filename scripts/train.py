import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

# MODEL_PATH = os.path.join(ROOT_DIR, "wandb_models", "model.zip")
# RB_PATH = os.path.join(ROOT_DIR, "wandb_models", "buffer.pkl")
print("PYTHONPATH updated:", ROOT_DIR)

import multiprocessing as mp
import gymnasium as gym
import utils.register_env as register_env
import wandb
import time
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from interactions.custom_encoder import CustomEncoder
from wandb.integration.sb3 import WandbCallback
from utils.wandb_callback import WandbCustomCallback
from loguru import logger

def train():
    run = wandb.init(
        project="offroad-sac-ds",
        # name=run_name,
        sync_tensorboard=True,
        dir="./",
        config={
            "master_seed": 0,
            "exec_mode": "common",
            "obs_mode": "full",

            "dem_id": "00000",

            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "batch_size": 512,
            "gamma": 0.99,
            "tau": 0.005,
            "total_timesteps": 50_0000,
            "n_envs": 16,
            "gradient_steps": 4,
            "train_freq": 1,
            "learning_starts": 5000,
            "features_dim": 256,

            "distance_w": 0.5,
            "yaw_w": 1.0,
            "collision_w": 1.0,
            "step_penalty_w": 1.0,
            "reach_w": 0.05,
            "risk_penalty_w": 0.2,
            "visit_penalty_w": 0.5,
            "nopath_w": 0.5,
            "explore_gain_w": 0.5,
        }
    )

    cfg = wandb.config

    logger.info(
        f"cfg.buffer_size={cfg.buffer_size}, cfg.batch_size={cfg.batch_size}, "
        f"cfg.n_envs={cfg.n_envs}, cfg.learning_rate={cfg.learning_rate}"
    )
    logger.info(f"cfg.total_timesteps={cfg.total_timesteps}, cfg.features_dim={cfg.features_dim}")


    run_name = (
        f"DM_dem{cfg.dem_id}_{cfg.exec_mode}_{cfg.obs_mode}"
        f"_seed{cfg.master_seed}"
        f"_env{cfg.n_envs}"
        f"_lr{float(cfg.learning_rate):.2e}"
        f"_bs{int(cfg.batch_size)}"
        f"_{time.strftime('%m%d-%H%M')}"
    )
    wandb.run.name = run_name

    EXEC_MODE = str(cfg.exec_mode)
    OBS_MODE = str(cfg.obs_mode)
    MASTER_SEED = int(cfg.master_seed)

    # ---dir----
    sweep_dir = os.path.join(ROOT_DIR, "sweep_runs")
    run_dir = os.path.join(sweep_dir, "_run_", wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)
    MODEL_PATH = os.path.join(run_dir, "model.zip")
    RB_PATH = os.path.join(run_dir, "buffer.pkl")

    # ---create multi-env---
    def make_env(rank: int):
        '''
        return: init()
        '''
        def _init():
            import gymnasium as gym    
            import utils.register_env
            from stable_baselines3.common.monitor import Monitor
            from interactions.attachments import param

            param.DISTANCE_WEIGHT = float(env_cfg["distance_w"])
            param.YAW_WEIGHT = float(env_cfg["yaw_w"])
            param.COLLISION_WEIGHT = float(env_cfg["collision_w"])
            param.STEP_PENALTY_WEIGHT = float(env_cfg["step_penalty_w"])
            param.REACH_GOAL_WEIGHT = float(env_cfg["reach_w"])
            param.RISK_PENALTY_WEIGHT = float(env_cfg["risk_penalty_w"])
            param.VISIT_PENALTY_WEIGHT = float(env_cfg["visit_penalty_w"])
            param.NOPATH_PENALTY_WEIGHT = float(env_cfg["nopath_w"])
            param.EXPLORE_GAIN_WEIGHT = float(env_cfg["explore_gain_w"])
            param.dem_id = str(env_cfg["dem_id"])
            # env.unwrapped.EXPLORE_GAIN_WEIGHT = float(env_cfg["explore_gain_w"])

            env = gym.make("DM-v1", exec_mode = EXEC_MODE, obs_mode = OBS_MODE, rank = rank)
            env = Monitor(env)

            return env
        return _init
    
    env_cfg = {
        "exec_mode": EXEC_MODE,
        "obs_mode": OBS_MODE,

        "dem_id":cfg.dem_id,

        "distance_w": cfg.distance_w,
        "yaw_w": cfg.yaw_w,
        "collision_w": cfg.collision_w,
        "step_penalty_w": cfg.step_penalty_w,
        "reach_w": cfg.reach_w,
        "risk_penalty_w": cfg.risk_penalty_w,
        "visit_penalty_w": cfg.visit_penalty_w,
        "nopath_w": cfg.nopath_w,
        "explore_gain_w": cfg.explore_gain_w,
    }

    env = SubprocVecEnv([make_env(i) for i in range(cfg.n_envs)], start_method="forkserver")
    env.seed(MASTER_SEED)

    # ---model create/read---
    policy_kwargs = dict(
        features_extractor_class=CustomEncoder,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256])
        )
        
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading weights from: {MODEL_PATH}")
        model = SAC.load(MODEL_PATH, env=env, device="cuda")
        if os.path.exists(RB_PATH): 
            model.load_replay_buffer(RB_PATH)
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            device="cuda",
            learning_rate=float(cfg.learning_rate),
            buffer_size=int(cfg.buffer_size),
            batch_size=int(cfg.batch_size),
            tau=float(cfg.tau),
            gamma=float(cfg.gamma),
            train_freq=(int(cfg.train_freq), "step"),
            gradient_steps=int(cfg.gradient_steps),
            ent_coef="auto",
            target_entropy="auto",
            policy_kwargs=policy_kwargs,
            verbose=2,
            tensorboard_log="./tb_logs/",
            learning_starts=int(cfg.learning_starts),
        )
    
    # ---callback---
    custom_cb = WandbCustomCallback(save_freq=5000, 
                                    verbose=2, 
                                    model_path=MODEL_PATH, 
                                    rb_path=RB_PATH,
                                    log_freq=1000)

    # ---learning---
    model.learn(
        total_timesteps=int(cfg.total_timesteps),
        log_interval=4,
        callback=[custom_cb]
    )

    # ---save---
    model.save(MODEL_PATH)
    model.save_replay_buffer(RB_PATH)
    env.close()
    run.finish()

if __name__ == "__main__":
    train()
