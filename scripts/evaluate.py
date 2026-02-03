import os
import sys
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from loguru import logger

# FILE_PATH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(ROOT_DIR)

import utils.register_env as register_env  
from interactions.attachments import param  

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def now_str():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def now_date():
    return time.strftime("%Y%m%d", time.localtime())

def jsonable(x: Any):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64, np.float16)):
        return float(x)
    if isinstance(x, (np.int32, np.int64, np.int16, np.uint8)):
        return int(x)
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x

class Agent:
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

class SB3Agent(Agent):
    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action

class RandomAgent(Agent):
    def __init__(self, action_space: gym.Space, seed: int = 0):
        self.action_space = action_space
        self.action_space.seed(seed)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return self.action_space.sample()

class FixedActionAgent(Agent):
    """
    Frontier/Greedy 

    fixed_action=np.array([0.5, 0.0, -0.2])

    action_space = gym.spaces.Box(
        low=-1.0, 
        high=1.0, 
        shape=(3,),  # 3个关节
        dtype=np.float32
    )
    """
    def __init__(self, fixed_action: np.ndarray, action_space: gym.Space):
        self.a = np.array(fixed_action, dtype=np.float32)
        self.action_space = action_space

        assert self.a.shape == self.action_space.shape, \
            f"fixed_action shape {self.a.shape} != action_space {self.action_space.shape}"

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        # 如果 action_space 是 Box，建议 clip 到合法范围
        if isinstance(self.action_space, gym.spaces.Box):
            low, high = self.action_space.low, self.action_space.high
            return np.clip(self.a, low, high)
        return self.a

class AblationWrapper(gym.Wrapper):
    """
    without_uncertainty:
      - 把 obs 里的 uncertainty 通道置零（或从 info/cost 中关闭相关项：看你 env 实现）
    without_planning:
      - 这种一般要在 env 内部切 planner 开关；如果 env 暴露了 set_mode/flags 最好
      - wrapper 里做不了规划器替换的话，就在 env 构造时传参数：gym.make(..., w/o_planning=True)
    """
    def __init__(self, env, exec_mode, obs_mode):
        super().__init__(env)
        self.exec_mode = exec_mode
        self.obs_mode = obs_mode

        self._apply_modes()

    def reset(self, **kwargs):
        self._apply_modes()

        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def _apply_modes(self):
        base = self.unwrapped 
        if self.exec_mode is not None:
            if not hasattr(base, "exec_mode"):
                raise AttributeError("env has no attribute 'exec_mode'")
            base.exec_mode = self.exec_mode

        if self.obs_mode is not None:
            if not hasattr(base, "obs_mode"):
                raise AttributeError("env has no attribute 'obs_mode'")
            base.obs_mode = self.obs_mode

@dataclass
class EpisodeSummary:
    ep_id: int
    seed: int
    steps: int
    episode_return: float
    success: bool
    collision: bool
    terminated: bool
    truncated: bool

class Recorder:
    """
    每条 episode 的记录器:保存轨迹、动作、奖励、info 关键字段
    """
    def __init__(self):
        self.poses: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.info_keys: List[Dict[str, Any]] = []
        self.success = False
        self.collision = False

        # fmt----------------------need complete
        self.m_occ = None
        self.m_free = None
        self.m_unk = None
        self.global_mask = None

    def append(self, obs: Dict[str, Any], action: np.ndarray, reward: float, info: Dict[str, Any]):
        self.poses.append(np.array(obs["pose"], dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(float(reward))

        key_info = {
            "success": info.get("success", None),
            "collision": info.get("collision", None),
            "done_reason": info.get("done_reason", None),
        }
        self.info_keys.append(key_info)

        self.success = self.success or bool(info.get("success", False))
        self.collision = self.collision or bool(info.get("collision", False))

    def summary(self, ep_id: int, seed: int, terminated: bool, truncated: bool) -> EpisodeSummary:
        return EpisodeSummary(
            ep_id=ep_id,
            seed=seed,
            steps=len(self.rewards),
            episode_return=float(np.sum(self.rewards)) if self.rewards else 0.0,
            success=bool(self.success),
            collision=bool(self.collision),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

def evaluate(
    env_id: str,
    agent: Agent,
    out_dir: str,
    n_episodes: int = 50,
    max_steps: int = 2000,
    ablation = "full",
    seed_base: int = 0,
    render: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---config20260101.yaml---
    config = dict(
        env_id=env_id,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed_base=seed_base,
        render=render,
        time=now_str(),
        ablation = ablation,
    )

    time_stamp = now_date()
    config_path = os.path.join(out_dir, f"config{time_stamp}.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # ---make env---
    env = gym.make(env_id)

    if ablation == "full":
        exec_mode, obs_mode = "common", "full"
    elif ablation == "without_uncertainty":
        exec_mode, obs_mode = "common", "no_uncertainty"
    elif ablation == "without_planning":
        exec_mode, obs_mode = "noplan", "full"
    else:
        raise ValueError(ablation)
    
    env = AblationWrapper(env, exec_mode=exec_mode, obs_mode=obs_mode)

    # ---main loop---
    summaries: List[EpisodeSummary] = []

    # ---episode---
    for ep in range(n_episodes):
        ep_seed = seed_base + ep
        seed_everything(ep_seed)

        obs, info = env.reset(seed=ep_seed)
        rec = Recorder()

        terminated = False
        truncated = False

        # ---step---
        for t in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()

            rec.append(obs, action, reward, info)

            # log
            if t % 50 == 0:
                pose = obs.get("pose", None)
                if pose is not None:
                    logger.info("ep={} t={} pose={} r={:.3f}", ep, t, pose, float(reward))
                else:
                    logger.info("ep={} t={} r={:.3f}", ep, t, float(reward))

            if terminated or truncated:
                break

        summ = rec.summary(ep_id=ep, seed=ep_seed, terminated=terminated, truncated=truncated)
        summaries.append(summ)

        # ---save one episodes---
        ep_dir = os.path.join(out_dir, f"ep_{ep:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        # summary --one episode
        with open(os.path.join(ep_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(summ), f, ensure_ascii=False, indent=2)

        # traj --one episode
        np.savez_compressed(
            os.path.join(ep_dir, "traj.npz"),
            poses=np.array(rec.poses, dtype=np.float32) if rec.poses else np.zeros((0,)),
            actions=np.array(rec.actions, dtype=np.float32) if rec.actions else np.zeros((0,)),
            rewards=np.array(rec.rewards, dtype=np.float32),
        )
        # info --one episode
        with open(os.path.join(ep_dir, "info_keys.json"), "w", encoding="utf-8") as f:
            json.dump(jsonable(rec.info_keys), f, ensure_ascii=False)

        # log
        logger.info(
            "EP DONE | ep={} seed={} steps={} return={:.3f} success={} collision={} term={} trunc={}",
            summ.ep_id, summ.seed, summ.steps, summ.episode_return,
            summ.success, summ.collision, summ.terminated, summ.truncated
        )

    env.close()

    # ---save one evaluate---
    # summarys --one evaluate
    with open(os.path.join(out_dir, "all_summaries.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, ensure_ascii=False, indent=2)

    succ_rate = float(np.mean([s.success for s in summaries])) if summaries else 0.0
    col_rate = float(np.mean([s.collision for s in summaries])) if summaries else 0.0
    avg_len = float(np.mean([s.steps for s in summaries])) if summaries else 0.0
    avg_ret = float(np.mean([s.episode_return for s in summaries])) if summaries else 0.0

    report = dict(
        n=len(summaries),
        success_rate=succ_rate,
        collision_rate=col_rate,
        avg_steps=avg_len,
        avg_return=avg_ret,
    )
    # save indicator --one evaluate
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # log
    logger.info("REPORT: {}", report)

def main():
    MASTER_SEED = 0

    # directory
    out_root = os.path.join(ROOT_DIR, "results_eval")
    run_dir = os.path.join(out_root, f"run_{now_str()}")
    os.makedirs(run_dir, exist_ok=True)

    # logger info
    logger.add(os.path.join(run_dir, "eval.log"), rotation="20 MB", retention="7 days")

    # obtain action_space
    base_env = gym.make("DM-v1")
    action_space = base_env.action_space
    base_env.close()

    # ---choose agent---
    METHOD = "sac"  # "sac" / "td3" / "random" / "frontier" / "greedy_goal"
    DETERMINISTIC = True

    if METHOD in ["sac", "td3"]:
        from stable_baselines3 import SAC, TD3
        env_for_model = gym.make("DM-v1")

        try:
            if METHOD == "sac":
                model = SAC.load("sac_decision_making", env=env_for_model)
            elif METHOD == "td3":
                model = TD3.load("td3_decision_making", env=env_for_model)
            else:
                raise ValueError(METHOD)
        finally:
                env_for_model.close()

        agent = SB3Agent(model, deterministic=DETERMINISTIC)

    elif METHOD == "random":
        agent = RandomAgent(action_space, seed = MASTER_SEED)

    elif METHOD == "frontier":
        # 这里假设你的 action = [w_goal, w_info, w_safe, tau]
        # frontier baseline 可以固定 w_info=1，其他=0
        agent = FixedActionAgent(
            fixed_action=np.array([0.5, 0.5, 0.5], dtype=np.float32),
            action_space=action_space,
        )

    elif METHOD == "greedy_goal":
        # greedy-goal：只追 goal（示例）
        agent = FixedActionAgent(
            fixed_action=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            action_space=action_space,
        )
    else:
        raise ValueError(f"Unknown METHOD: {METHOD}")

    # ---choose ablation way---
    '''
    without_uncertainty:exec(common), obs(no_uncertainty);
    without_planning:exec(noplan), obs(full);
    full:exec(common), obs(full);
    '''
    ABLATION = "without_uncertainty"      # without_planning, full

    # ---evaluate---
    evaluate(
        env_id="DM-v1",
        agent=agent,
        out_dir=os.path.join(run_dir, f"{METHOD}_{ABLATION}"),
        n_episodes=50,
        max_steps=2000,
        ablation = ABLATION,
        seed_base=MASTER_SEED,
        render=False,
    )

if __name__ == "__main__":
    main()
