import gymnasium as gym
from gymnasium import spaces
import numpy as np  
from lidar import Lidar
from read_grid_map import ReadGridMap


class DM_env(gym.Env):
    def __init__(self):
        super(DM_env, self).__init__()

        self.local_size = 64  # 局部地图大小


        # 定义动作空间：
        '''
        子目标点(x,y)
        '''
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), # sub_goal_x/100, sub_goal_y/100
            high=np.array([1.0, 1.0]), # sub_goal_x/100, sub_goal_y/100
            shape=(2,),
            dtype=float
        ) 

        # 定义观测空间：
        '''
        智能体坐标(x,y)
        目标点坐标(x_goal,y_goal)
        map(uncertainty map)二维数组展平
        map(occupancy grid map)二维数组展平
        '''
        self.observation_space = spaces.Dict({
            "map": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2, self.local_size, self.local_size),
                dtype=np.float32
            ),

            "pose": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),   # [x, y, gx, gy]
                dtype=np.float32
            )
        })


        # self.observation_space = spaces.Box(
        #     low=np.array([-np.inf, -np.inf]),
        #     high=np.array([np.inf, np.inf]),
        #     shape=(2,),
        #     dtype=float
        # )

    def reset(self):
        # 重置环境状态
        observation = np.array([0.0, 0.0])  # 初始观测值
        return observation, {}

    def step(self, action):
        # 执行动作，更新环境状态
        observation = np.array([0.0, 0.0])  # 新的观测值
        reward = 0.0  # 奖励值
        done = False  # 是否结束
        info = {}  # 额外信息
        return observation, reward, done, False, info

    def render(self, mode='human'):
        # 渲染环境（可选）
        pass    

   

