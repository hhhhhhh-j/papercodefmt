import gymnasium as gym
from gymnasium.envs.registration import register
register(
    id="DM-v0",                      # 你 environment 的名字
    entry_point="SAC_DM:DM_env",     # 文件名:类名
)