import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np  
import math



def generate_random_map(obstacle_ratio=0.2, width=256, height=256, seed = None):
    """
    使用 gym 环境的随机系统生成随机地图
    1 = obstacle
    0 = free
    """

    np_random, _ = seeding.np_random(seed)

    random_matrix = np_random.random((height, width))
    global_map = (random_matrix < obstacle_ratio).astype(np.uint8)

    return global_map

def main_map():
    map = generate_random_map()
    print(map)

if __name__ == "__main__":
    