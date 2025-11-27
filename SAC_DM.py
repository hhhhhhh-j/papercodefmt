import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np  
import math
from train_env import Lidar
from train_env import param
from read_grid_map import ReadGridMap # 读取真实栅格地图
from train_env import interface2RL





class DM_env(gym.Env):
    def __init__(self):
        super(DM_env, self).__init__()
        # 环境参数
        self.step_count = 0
        self.np_random, _ = seeding.np_random(None)
        self.agent_x = 0
        self.agent_y = 0
        self.agent_yaw = 0
        self.goal_x = 0
        self.goal_y = 0
        self.goal_yaw = 0
        self.width = 0
        self.height = 0
        # map
        self.global_map = np.zeros((self.height, self.width), dtype=np.uint8)
        self.map_uncertainty_local
        self.map_occupancy_local
        # 缩放系数
        self.action_ratio # 动作缩放系数
        self.observation_ratio_xy # 观测空间xy缩放系数
        self.observation_ratio_yaw # 观测空间yaw缩放系数

        # 创建调用对象
        self.interface = interface2RL(self.map)
        self.lidar = Lidar(self.map)


        # 定义动作空间：
        '''
        子目标点(x,y)
        '''
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), # sub_goal_x/100, sub_goal_y/100, yaw/2pi
            high=np.array([1.0, 1.0, 1.0]), # sub_goal_x/100, sub_goal_y/100, yaw/2pi
            shape=(3,),
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
                shape=(2, param.local_size_height, param.local_size_width),  # occupancy map + uncertainty map
                dtype=np.float32
            ),

            "pose": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(6,),   # [x, y, yaw, gx, gy, gyaw]
                dtype=np.float32
            )
        })

        # # 定义状态空间
        # self.state = {
        #     "map": self.global_map,
        #     "pose": np.array([
        #         self.agent_x,
        #         self.agent_y,
        #         self.agent_yaw,
        #         self.goal_x,
        #         self.goal_y,
        #         self.goal_yaw
        #     ])
        # }

    def reset(self): # bugggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
        '''
        -重置环境状态
        -返回值
        observation	    初始观察
        info	        额外信息（一般空字典）
        '''

        self.step_count = 0

        # 随机生成地图
        self.generate_random_map(seed = None) # 随机生成地图 -self.global_map

        # 随机生成智能体和目标点位置（注意要避开障碍物）
        self.agent_x,self.agent_y,self.agent_yaw = self.get_random_free_position() 
        self.goal_x,self.goal_y,self.goal_yaw = self.get_random_free_position() 

        # 获取初始观测值（map_uncertainty、map_occupancy）
        lidar = interface2RL()
        map_uncertainty_local = lidar.get_local_uncertainty_map() # buggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
        map_occupancy_local = lidar.get_local_occupy_map() # bugggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg

        # 写入观测值
        observation = {
            "map": np.stack([map_occupancy_local, map_uncertainty_local], axis=0).astype(np.float32),

            "pose": np.array([
                self.agent_x / self.width,
                self.agent_y / self.height,
                self.agent_yaw / (2 * math.pi),
                self.goal_x / self.width,
                self.goal_y / self.height,
                self.goal_yaw / (2 * math.pi)
            ], dtype=np.float32)
        }
        
        return observation, {}

    def step(self, action): # bugggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
        '''
        返回值
        observation	    observation	        下一时刻的观测
        reward	        reward	            这一步的奖励
        done	        terminated	        任务是否自然结束（成功/失败）
        False	        truncated	        是否被强制中断（超时等）
        info	        info	            额外信息（调试/分析用）
        '''
        # 执行动作，更新环境状态
        self.step_count += 1
        return observation, reward, done, False, info
    

    def render(self, mode='human'):
        '''
        渲染环境（可选）
        mode: 渲染模式
        'human'：渲染到屏幕
        'rgb_array'：返回RGB图像数组
        '''
        pass    

    def generate_random_map(self, obstacle_ratio=0.2, width=256, height=256, seed = None):
        """
        使用 gym 环境的随机系统生成随机地图
        1 = obstacle
        0 = free
        """
        self.width = width
        self.height = height
        self.np_random, _ = seeding.np_random(seed)

        random_matrix = self.np_random.random((height, width))
        self.global_map = (random_matrix < obstacle_ratio).astype(np.uint8)

    def get_random_free_position(self):
        '''
        获取随机的空闲位置 (x, y, yaw)
        '''
        free_cells = np.argwhere(self.global_map == 0)
        idx = self.np_random.integers(0, len(free_cells))
        y, x = free_cells[idx]
        yaw = self.np_random.uniform(-np.pi, np.pi)

        return x, y, yaw
            

if __name__ == "__main__":
    pass
   

