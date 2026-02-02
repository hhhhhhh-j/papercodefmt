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

# ====== 你的工程路径处理（保持你原来的写法即可）======
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(ROOT_DIR)

import utils.register_env as register_env  # noqa: F401  确保 DM-v1 被注册
# from interactions.attachments import param  # 如果你需要 param 里的一些常量可打开


# -----------------------------
# 1) 可复现：统一的 seed 设置
# -----------------------------
def seed_everything(seed: int):
    """确保随机性可控：python / numpy（torch 如果你用到再加）"""
    random.seed(seed)
    np.random.seed(seed)

    # 如果你的评测里用到了 torch（比如某些手写 policy 用 torch），加上：
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def now_str():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def jsonable(x: Any):
    """把 numpy 类型转成可 JSON 序列化的形式（否则 json.dump 可能报错）"""
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


# -----------------------------
# 2) 统一接口：所有方法都是一个 Agent
# -----------------------------
class Agent:
    """统一接口：act(obs)->action"""
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError


class SB3Agent(Agent):
    """SAC / TD3 都能用这个封装，只要传入 SB3 模型即可"""
    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


class RandomAgent(Agent):
    """随机 baseline：严格从 env.action_space 采样（接口完全一致）"""
    def __init__(self, action_space: gym.Space, seed: int = 0):
        self.action_space = action_space
        self.action_space.seed(seed)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return self.action_space.sample()


# 注意：frontier / greedy 这类非学习 baseline
# 最“论文友好”的方式是：它们也输出同样的 action 向量
# 例如你的 action 是 [w_goal, w_info, w_safe, tau] 这样的权重/温度参数
# 那 frontier 就固定输出 w_info=1；greedy-goal 输出 w_goal=1 等
class FixedActionAgent(Agent):
    """
    用固定动作向量实现 Frontier/Greedy 的“同接口”版本

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


# -----------------------------
# 3) 消融：用 Wrapper 控制“关模块但不改接口”
# -----------------------------
class AblationWrapper(gym.Wrapper):
    """
    without_uncertainty:
      - 把 obs 里的 uncertainty 通道置零（或从 info/cost 中关闭相关项：看你 env 实现）
    without_planning:
      - 这种一般要在 env 内部切 planner 开关；如果 env 暴露了 set_mode/flags 最好
      - wrapper 里做不了规划器替换的话，就在 env 构造时传参数：gym.make(..., w/o_planning=True)
    """
    def __init__(self, env, without_uncertainty: bool = False):
        super().__init__(env)
        self.without_uncertainty = without_uncertainty

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._process_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(obs)
        return obs, reward, terminated, truncated, info

    def _process_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.without_uncertainty:
            return obs

        # 例子：如果你 obs 里有 "global_map" 的某个通道是 uncertainty
        # 你需要按你自己的 obs 结构改这里
        # 常见：global_map shape = (C,H,W)，其中某个 channel 表示 uncertainty/unknown
        if "global_map" in obs:
            gm = np.array(obs["global_map"])
            # 举例：假设第 2 个通道是 uncertainty（你需要改成真实索引）
            if gm.ndim == 3 and gm.shape[0] >= 3:
                gm[2, :, :] = 0.0
                obs["global_map"] = gm

        # 例子：如果你单独给了 "uncertainty_map"
        if "uncertainty_map" in obs:
            obs["uncertainty_map"] = np.zeros_like(obs["uncertainty_map"])

        return obs


# -----------------------------
# 4) 记录与汇总：论文主表最小指标
# -----------------------------
@dataclass
class EpisodeSummary:
    ep_id: int
    seed: int
    steps: int
    return_sum: float
    success: bool
    collision: bool
    terminated: bool
    truncated: bool


class Recorder:
    """每条 episode 的记录器：保存轨迹、动作、奖励、info 关键字段"""
    def __init__(self):
        self.poses: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.info_keys: List[Dict[str, Any]] = []

        self.success = False
        self.collision = False

    def append(self, obs: Dict[str, Any], action: np.ndarray, reward: float, info: Dict[str, Any]):
        if "pose" in obs:
            self.poses.append(np.array(obs["pose"], dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(float(reward))

        # 只存 info 的关键字段（避免 info 里塞大数组导致文件爆炸）
        key_info = {
            "success": info.get("success", None),
            "collision": info.get("collision", None),
            "done_reason": info.get("done_reason", None),
        }
        self.info_keys.append(key_info)

        # success/collision 的 key 你要按你 env 实际字段改
        self.success = self.success or bool(info.get("success", False))
        self.collision = self.collision or bool(info.get("collision", False))

    def summary(self, ep_id: int, seed: int, terminated: bool, truncated: bool) -> EpisodeSummary:
        return EpisodeSummary(
            ep_id=ep_id,
            seed=seed,
            steps=len(self.rewards),
            return_sum=float(np.sum(self.rewards)) if self.rewards else 0.0,
            success=bool(self.success),
            collision=bool(self.collision),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )


# -----------------------------
# 5) Runner：批量跑、每条落盘、汇总输出
# -----------------------------
def evaluate(
    env_id: str,
    agent: Agent,
    out_dir: str,
    n_episodes: int = 50,
    max_steps: int = 2000,
    seed_base: int = 0,
    render: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    # 写入 config：论文复现必备
    config = dict(
        env_id=env_id,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed_base=seed_base,
        render=render,
        time=now_str(),
    )
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 注意：环境建议每次 evaluate 构造一次即可
    env = gym.make(env_id)

    summaries: List[EpisodeSummary] = []

    for ep in range(n_episodes):
        ep_seed = seed_base + ep
        seed_everything(ep_seed)

        obs, info = env.reset(seed=ep_seed)
        rec = Recorder()

        terminated = False
        truncated = False

        for t in range(max_steps):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if render:
                env.render()

            rec.append(obs, action, reward, info)

            # 日志：论文跑大批量时建议写入文件，别刷屏
            # 这里给你最简可读的
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

        # 每条 episode 单独落盘：防止中途崩掉全没
        ep_dir = os.path.join(out_dir, f"ep_{ep:04d}")
        os.makedirs(ep_dir, exist_ok=True)

        with open(os.path.join(ep_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(summ), f, ensure_ascii=False, indent=2)

        # 保存轨迹（npz 最合适）
        np.savez_compressed(
            os.path.join(ep_dir, "traj.npz"),
            poses=np.array(rec.poses, dtype=np.float32) if rec.poses else np.zeros((0,)),
            actions=np.array(rec.actions, dtype=np.float32) if rec.actions else np.zeros((0,)),
            rewards=np.array(rec.rewards, dtype=np.float32),
        )

        with open(os.path.join(ep_dir, "info_keys.json"), "w", encoding="utf-8") as f:
            json.dump(jsonable(rec.info_keys), f, ensure_ascii=False)

        logger.info(
            "EP DONE | ep={} seed={} steps={} return={:.3f} success={} collision={} term={} trunc={}",
            summ.ep_id, summ.seed, summ.steps, summ.return_sum,
            summ.success, summ.collision, summ.terminated, summ.truncated
        )

    env.close()

    # 汇总文件：主表数据来源
    with open(os.path.join(out_dir, "all_summaries.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, ensure_ascii=False, indent=2)

    # 控制台快速报告：论文主表最小指标
    succ_rate = float(np.mean([s.success for s in summaries])) if summaries else 0.0
    col_rate = float(np.mean([s.collision for s in summaries])) if summaries else 0.0
    avg_len = float(np.mean([s.steps for s in summaries])) if summaries else 0.0
    avg_ret = float(np.mean([s.return_sum for s in summaries])) if summaries else 0.0

    report = dict(
        n=len(summaries),
        success_rate=succ_rate,
        collision_rate=col_rate,
        avg_steps=avg_len,
        avg_return=avg_ret,
    )
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("REPORT: {}", report)


# -----------------------------
# 6) main：选择方法（SAC/TD3/Random/Frontier/Greedy）+ 消融开关
# -----------------------------
def main():
    # 结果目录：每次运行一个独立文件夹，天然可追溯
    out_root = os.path.join(ROOT_DIR, "results_eval")
    run_dir = os.path.join(out_root, f"run_{now_str()}")
    os.makedirs(run_dir, exist_ok=True)

    # 建议把日志也落盘
    logger.add(os.path.join(run_dir, "eval.log"), rotation="20 MB", retention="7 days")

    # ====== 构造 env（先构造一份用于拿 action_space）======
    base_env = gym.make("DM-v1")
    action_space = base_env.action_space
    base_env.close()

    # ====== 选择 agent ======
    METHOD = "sac"  # "sac" / "td3" / "random" / "frontier" / "greedy_goal"
    DETERMINISTIC = True

    if METHOD in ["sac", "td3"]:
        from stable_baselines3 import SAC, TD3
        env_for_model = gym.make("DM-v1")

        if METHOD == "sac":
            model = SAC.load("sac_decision_making", env=env_for_model)
        else:
            model = TD3.load("td3_decision_making", env=env_for_model)

        agent = SB3Agent(model, deterministic=DETERMINISTIC)

    elif METHOD == "random":
        agent = RandomAgent(action_space, seed=0)

    elif METHOD == "frontier":
        # 这里假设你的 action = [w_goal, w_info, w_safe, tau]
        # frontier baseline 可以固定 w_info=1，其他=0
        agent = FixedActionAgent(
            fixed_action=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            action_space=action_space,
        )

    elif METHOD == "greedy_goal":
        # greedy-goal：只追 goal（示例）
        agent = FixedActionAgent(
            fixed_action=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            action_space=action_space,
        )
    else:
        raise ValueError(f"Unknown METHOD: {METHOD}")

    # ====== 消融开关（关键：接口不变，只改内部模块/观测）======
    WITHOUT_U = False
    # WITHOUT_PLANNING 一般需要 env 内部支持开关（wrapper 很难替换 planner）
    # 最好做成 gym.make("DM-v1", without_planning=True) 这种
    # 这里演示 without_uncertainty 用 wrapper

    # 注意：evaluate() 里自己 gym.make(env_id)
    # 若要用 wrapper，你有两种方式：
    # 1) 把 wrapper 逻辑搬到 evaluate() 里
    # 2) 或者让 DM-v1 支持参数 without_uncertainty=True
    #
    # 这里我给你建议：让你的 env 注册支持 kwargs：
    # gym.register(..., kwargs={"without_uncertainty": True})
    # 或 gym.make("DM-v1", without_uncertainty=True)
    #
    # 如果你暂时做不到，那就把 evaluate() 的 env = gym.make(...) 改成下面这种：
    # env = AblationWrapper(gym.make(env_id), without_uncertainty=WITHOUT_U)

    # ====== 跑评测 ======
    evaluate(
        env_id="DM-v1",
        agent=agent,
        out_dir=os.path.join(run_dir, f"{METHOD}_woU{int(WITHOUT_U)}"),
        n_episodes=50,
        max_steps=2000,
        seed_base=0,
        render=False,
    )


if __name__ == "__main__":
    main()
