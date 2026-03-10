import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

import gymnasium as gym
import utils.register_env as register_env
import numpy as np
from stable_baselines3 import SAC
from loguru import logger
from interactions.attachments import param


def angle_wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def main():
    env = gym.make("DM-v1")
    # model = SAC.load("sac_decision_making", env=env)

    obs, info = env.reset()

    for t in range(1):
        # action, _ = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        action = [0.3, 0.0, 0.0]  # dummy action for testing
        obs, reward, terminated, truncated, info = env.step(action)
        # env.render()

        # --- 1) 必要字段检查（你需要先在 env 里按 A 加入 info） ---
        need = ["agent_ix","agent_iy","goal_ix","goal_iy","agent_x","agent_y","goal_x","goal_y","agent_yaw","goal_angle"]
        if not all(k in info for k in need):
            logger.warning("Missing debug keys in info. Please add them in env.step(): {}", [k for k in need if k not in info])
            break

        agent_ix, agent_iy = info["agent_ix"], info["agent_iy"]
        goal_ix, goal_iy = info["goal_ix"], info["goal_iy"]
        agent_x, agent_y = info["agent_x"], info["agent_y"]
        goal_x, goal_y = info["goal_x"], info["goal_y"]
        yaw = info["agent_yaw"]
        goal_angle = info["goal_angle"]

        # --- 2) world<->index round-trip（验证你的 Transform_* 是否闭环） ---
        # 这里假设 res=1；如果 res!=1，也照样成立（只是值尺度不同）
        # 如果你采用的是“world y向上、index y向下”的定义：
        H = param.global_size_height
        res = param.XY_RESO
        ix2 = int(round(agent_x / res))
        iy2 = int(round((H - 1) - (agent_y / res)))

        if abs(ix2 - agent_ix) > 1 or abs(iy2 - agent_iy) > 1:
            logger.error("Index<->World mismatch! idx=({},{}), world=({:.3f},{:.3f}), idx2=({},{})",
                         agent_ix, agent_iy, agent_x, agent_y, ix2, iy2)
            break

        # --- 3) 角度一致性：用 index 坐标系计算 goal_angle_idx，看是否接近 env 给的 goal_angle ---
        # interface2RL 明确是 index 坐标系（y向下），所以 atan2(dy,dx) 应该用 (goal_iy-agent_iy)
        ga_idx = float(np.arctan2(goal_iy - agent_iy, goal_ix - agent_ix))
        diff = angle_wrap(goal_angle - ga_idx)

        if abs(diff) > 0.3:  # 0.3rad~17deg，你可以先宽松点
            logger.warning("goal_angle seems inconsistent: env={:.3f}, idx_calc={:.3f}, diff={:.3f}",
                           goal_angle, ga_idx, diff)

        if t % 20 == 0:
            logger.info("t={} idx=({},{}) world=({:.1f},{:.1f}) goal_idx=({},{}) r={:.3f}",
                        t, agent_ix, agent_iy, agent_x, agent_y, goal_ix, goal_iy, reward)

        if terminated or truncated:
            logger.info("Episode finished: terminated={}, truncated={}, reason={}", terminated, truncated, info.get("done_reason"))
            break

    env.close()

if __name__ == "__main__":
    main()