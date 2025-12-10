import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class WandbCustomCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.successes = 0
        self.collisions = 0

    def _on_step(self) -> bool:
        super()._on_step()
        infos = self.locals["infos"]
        dones = self.locals["dones"]   # 关键：VecEnv 的 done 列表

        # 如果有 episode 结束
        if dones[0]:
            self.episode_count += 1
            info = infos[0]

            if info.get("reach_goal", False):
                self.successes += 1
            if info.get("collision", False):
                self.collisions += 1

        # 定期 log（例如每1000步）
        if self.num_timesteps % self.log_freq == 0:
            success_rate = self.successes / max(1, self.episode_count)
            collision_rate = self.collisions / max(1, self.episode_count)

            wandb.log({
                "custom/success_rate": success_rate,
                "custom/collision_rate": collision_rate,
                "custom/episodes": self.episode_count,
                "timesteps": self.num_timesteps,
            })

        return True
