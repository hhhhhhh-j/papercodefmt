import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class WandbCustomCallback(BaseCallback):
    def __init__(self, save_freq = 5000, verbose=2, model_path = None, rb_path = None, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.model_path = model_path
        self.rb_path = rb_path

        self.episode_count = 0
        self.successes = 0
        self.collisions = 0
        self._last_save = 0
        self.last_log = 0

    def _on_step(self) -> bool:
        super()._on_step()
        infos = self.locals["infos"]
        dones = self.locals["dones"]   

        # success and collision (遍历所有并行环境)
        for d, info in zip(dones, infos):
            if d:
                self.episode_count += 1
                if info.get("success", False):
                    self.successes += 1
                if info.get("collision", False):
                    self.collisions += 1


        # 定期 log
        if self.log_freq and (self.num_timesteps - self.last_log) >= self.log_freq:
            self.last_log = self.num_timesteps

            success_rate = self.successes / max(1, self.episode_count)
            collision_rate = self.collisions / max(1, self.episode_count)

            wandb.log({
                "custom/success_rate": success_rate,
                "custom/collision_rate": collision_rate,
                "custom/episodes": self.episode_count,
                "timesteps": self.num_timesteps,
            }, step=self.num_timesteps)
            
        # save
        if self.save_freq and (self.num_timesteps - self._last_save) >= self.save_freq:
            self._last_save = self.num_timesteps

            if self.model_path is not None:
                self.model.save(self.model_path)

            # if self.rb_path is None:
            #     self.model.save_replay_buffer(self.rb_path)

            wandb.log({"custom/checkpoint_saved_at": self.num_timesteps})

        return True
