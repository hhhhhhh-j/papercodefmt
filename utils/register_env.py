import gymnasium as gym
from gymnasium.envs.registration import register
register(
    id="DM-v0",                      # 你 environment 的名字
    entry_point="models.SAC_DM:DM_env",     # 文件名:类名
)
register(
    id="DM-v1",                      # 你 environment 的名字    0126 v2
    entry_point="models.SAC_DS_DM:DM_env",     # 文件名:类名
)
