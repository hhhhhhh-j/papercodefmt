import gymnasium as gym
from gymnasium import spaces
import numpy as np  
import math
from train_env import Lidar
from read_grid_map import ReadGridMap
from Get_map import Map
from train_env import interface2RL



class DM_env(gym.Env):
    def __init__(self):
        '''
        -定义动作空间
        -定义观测空间
        '''
        super(DM_env, self).__init__()

        self.local_size = 64  # 局部地图大小
        self.step_count = 0

        self.agent_x
        self.agent_y
        self.agent_yaw
        self.goal_x
        self.goal_y
        self.goal_yaw
        self.global_map
        self.map_uncertainty_local
        self.map_occupancy_local

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
                shape=(2, self.local_size, self.local_size),
                dtype=np.float32
            ),

            "pose": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(6,),   # [x, y, yaw, gx, gy, gyaw]
                dtype=np.float32
            )
        })

        # 定义状态空间
        self.state = {
            "map": self.global_map,
            "pose": np.array([
                self.agent_x,
                self.agent_y,
                self.agent_yaw,
                self.goal_x,
                self.goal_y,
                self.goal_yaw
            ])
        }

    def reset(self):
        '''
        -重置环境状态
        -返回值
        observation	    初始观察
        info	        额外信息（一般空字典）
        '''

        self.step_count = 0

        # 随机生成地图
        map = Map()
        self.global_map = map.generate_random_map() # bugggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
        (self.M, self.N) = np.shape(self.global_map)

        # 随机生成智能体和目标点位置（注意要避开障碍物）
        x,y,yaw = map.get_random_free_position() # buggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
        gx,gy,gyaw = map.get_random_free_position() # bugggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
        self.agent_x = x
        self.agent_y = y
        self.agent_yaw = yaw
        self.goal_x = gx
        self.goal_y = gy
        self.goal_yaw = gyaw

        # 获取初始观测值（map_uncertainty、map_occupancy）
        local_map = interface2RL()

        map_uncertainty_local = local_map.get_local_uncertainty_map() # buggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg
        map_occupancy_local = local_map.get_local_occupy_map() # bugggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg

        # 写入观测值
        observation = {
            "map": np.stack([map_occupancy_local, map_uncertainty_local], axis=0).astype(np.float32),

            "pose": np.array([
                self.agent_x / self.M,
                self.agent_y / self.N,
                self.agent_yaw / (2 * math.pi),
                self.goal_x / self.M,
                self.goal_y / self.N,
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

   

