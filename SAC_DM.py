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
        # 参数
        self.step_count = 0
        self.max_steps = 10000
        self.seed = param.SEED
        self.width = param.global_size_width
        self.height = param.global_size_height
        # action 缩放系数
        self.ratio_x = param.RATIO_x
        self.ratio_y = param.RATIO_y
        self.ratio_yaw = param.RATIO_yaw
        # 全局变量
        self.np_random, _ = seeding.np_random(self.seed)
        self.agent_x = 0
        self.agent_y = 0
        self.agent_yaw = 0
        self.goal_x = 0
        self.goal_y = 0
        self.goal_yaw = 0
        self.timestep = 0
        self.global_map = np.zeros((self.height, self.width), dtype=np.uint8)

        # 定义动作空间：
        '''
        子目标点(x,y)
        '''
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            shape=(3,),
            dtype=np.float32
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

    def reset(self, *, seed=None, options=None):
        '''
        -重置环境状态
        -返回值
        observation	    初始观察
        info	        额外信息（一般空字典）
        '''
        super().reset(seed=seed)

        self.step_count = 0

        # 生成地图
        # self.generate_random_map(seed = seed)     # 随机生成地图 
        self.get_easy_map()                         # 生成简单测试地图

        # 随机生成智能体和目标点位置（注意要避开障碍物）
        self.agent_x,self.agent_y,self.agent_yaw = self.get_random_free_position() 
        self.goal_x,self.goal_y,self.goal_yaw = self.get_random_free_position() 

        # 初始化创建接口对象
        self.interface = interface2RL(self.global_map, 
                                      [self.agent_x, self.agent_y, self.agent_yaw])
        
        # 获取初始观测值（map_uncertainty、map_occupancy）
        local_m, local_m_uncertainty, _ = self.interface.ToSAC_reset()
        
        # 写入观测值
        observation = {
            "map": np.stack([local_m, local_m_uncertainty], axis=0).astype(np.float32),

            "pose": np.array([
                self.agent_x / self.width,
                self.agent_y / self.height,
                self.agent_yaw / math.pi,
                self.goal_x / self.width,
                self.goal_y / self.height,
                self.goal_yaw / math.pi
            ], dtype=np.float32)
        }
        
        return observation, {}

    def step(self, action): 
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

        # 计算上一时刻与终点的distance和yaw
        distance_MH = abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)
        distance_Euclidean = math.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        yaw_diff = abs(self.agent_yaw - self.goal_yaw)
        yaw_diff = min(yaw_diff, 2*math.pi - yaw_diff)

        # 计算 action - sub goal
        sub_x = np.clip(self.agent_x + action[0] * self.ratio_x, 0, param.global_size_width)
        sub_y = np.clip(self.agent_y + action[1] * self.ratio_y, 0, param.global_size_height)
        sub_yaw = self.agent_yaw + action[2] * self.ratio_yaw
        sub_goal = [sub_x, sub_y, sub_yaw]
        
        # 更新局部map与pose
        local_m, local_m_uncertainty, current_pose, collision = self.interface.ToSAC_step(sub_goal)

        self.agent_x = current_pose[0]
        self.agent_y = current_pose[1]
        self.agent_yaw = current_pose[2]
        
        # -----填充observation-----
        observation = {
                    "map": np.stack([local_m, local_m_uncertainty], axis=0).astype(np.float32),

                    "pose": np.array([
                        self.agent_x / self.width,
                        self.agent_y / self.height,
                        self.agent_yaw / math.pi,
                        self.goal_x / self.width,
                        self.goal_y / self.height,
                        self.goal_yaw / math.pi
                    ], dtype=np.float32)
                }
        
        # -----reward-----
        # 计算当前时刻与终点的distance和yaw
        distance_MH_new = abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)
        distance_Euclidean_new = math.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        yaw_diff_new = abs(self.agent_yaw - self.goal_yaw)
        yaw_diff_new = min(yaw_diff_new, 2*math.pi - yaw_diff_new)

        # 不确定性增益（两种可选，后续可优化这部分）
        uncertain_gain = np.mean(local_m_uncertainty)
        # uncertain_gain = np.max(local_m_uncertainty)

        reach = self.reach_goal()

        # reward计算
        distance_reward = param.DISTANCE_WEIGHT * (distance_Euclidean - distance_Euclidean_new)
        yaw_reward = param.YAW_WEIGHT * (yaw_diff - yaw_diff_new)
        collision_penalty = param.COLLISION_PENALTY if collision else 0.0
        step_penalty = param.STEP_PENALTY
        reach_goal_reward = param.REACH_GOAL_REWARD if reach else 0.0
        move_reward = param.MOVE_STEP * np.linalg.norm(action[:2])
        explore_reward = param.EXPLORE_GAIN * uncertain_gain

        # 防止程序崩溃
        # 1. 防止距离坐标溢出
        distance_Euclidean = float(np.nan_to_num(distance_Euclidean, nan=500.0, posinf=500.0, neginf=500.0))
        distance_Euclidean_new = float(np.nan_to_num(distance_Euclidean_new, nan=500.0, posinf=500.0, neginf=500.0))

        # 2. 不确定性保证在 [0,1]
        uncertain_gain = float(np.nan_to_num(uncertain_gain, nan=1.0, posinf=1.0, neginf=1.0))
        uncertain_gain = max(0.0, min(1.0, uncertain_gain))

        # 3. 航向差限制在合法范围
        yaw_diff = float(np.nan_to_num(yaw_diff, nan=0.0))
        yaw_diff_new = float(np.nan_to_num(yaw_diff_new, nan=0.0))

        reward = (distance_reward + 
                  yaw_reward + 
                  collision_penalty + 
                  step_penalty + 
                  reach_goal_reward + 
                  explore_reward + 
                  move_reward)

        reward = np.clip(reward, -200, 200)

        # -----设置done和truncated-----
        done = True if (reach or collision) else False
        truncated = True if self.step_count >= self.max_steps else False

        # -----调试信息-----
        if reach:
            print("reachhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")

        if collision:
            print("collisionnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        
        print("current.pose:",self.agent_x,self.agent_y)
        print("goal:",self.goal_x,self.goal_y)
        print("reward:",reward)

        # -----输出info-----
        info = {
            "reach_goal": reach,
            "collision": collision,
            "distance_to_goal": distance_Euclidean_new,
            "uncertainty": np.mean(local_m_uncertainty)
        }

        self.timestep += 1
        print("step:",self.timestep)
        
        return observation, reward, done, truncated, info
    
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

    def get_easy_map(self, width=256, height=256):
        map_ = np.zeros((height, width), dtype=np.uint8)

        # 1. 四周边框
        map_[0, :] = 1
        map_[-1, :] = 1
        map_[:, 0] = 1
        map_[:, -1] = 1

        # 2. 中间横墙，高 8 像素，留左右两个大缺口
        wall_y = height // 2
        gap = width // 4          # 每个缺口宽度
        left_gap_start = gap
        right_gap_start = width - gap

        # 横墙主体
        map_[wall_y - 4 : wall_y + 4, left_gap_start + gap : right_gap_start] = 1
        self.global_map = map_

    def get_random_free_position(self):
        '''
        获取随机的空闲位置 (x, y, yaw)
        '''
        free_cells = np.argwhere(self.global_map == 0)
        idx = self.np_random.integers(0, len(free_cells))
        y, x = free_cells[idx]
        yaw = self.np_random.uniform(-np.pi, np.pi)

        return x, y, yaw
    
    def reach_goal(self):
        distance = math.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        yaw_diff = abs(self.agent_yaw - self.goal_yaw)
        yaw_diff = min(yaw_diff, 2*math.pi - yaw_diff)

        if distance < 5.0 and yaw_diff < (15.0 * math.pi / 180.0):
            return True
        else:
            return False
            
if __name__ == "__main__":
    '''
    待完善部分
    1.动作空间是否可以直接设计为 throttle，yaw
    2.地图：如何随机的生成合理（具备可行性，满足越野场景）的地图
    3.奖励函数目前不完善，可能有错误
    4.状态空间是否可以加入一些视觉的东西
    5.是否要用开源数据集去跑
    6.如何用render进行可视化
    7.后续是否要用pybullet来跑
    '''
    pass
   

