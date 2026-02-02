import sys, os
import gymnasium as gym
import utils.register_env as register_env
import numpy as np
from stable_baselines3 import SAC
from loguru import logger
from interactions.attachments import param

# ---dir----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

class Pack_param:
    def __init__(self):
        # param to deal  (metrics)
        H, W = param.global_size_height, param.global_size_width
        self.global_map = None
        self.process_occ_map = None
        self.process_free_map = None
        self.process_unk_map = None

        self.trajectory = []
        self.rewards = []
        self.collision_points = []

        self.start_pose = []
        self.goal_pose = []
        

    def read_data(self, infos):
        pass

    def save2file(self):
        pass

def main():
    pack = Pack_param()

    env = gym.make("DM-v1")
    model = SAC.load("sac_decision_making", env=env)

    obs, info = env.reset()

    for _ in range(1000):  
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        logger.info("pose:{}", obs["pose"], "reward:{}", reward)

        if terminated or truncated:
            logger.info("Episode finished")
            break

    env.close()

if __name__ == "__main__":
    main()
