import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np  
import math
from envs.env_Entropy import Lidar
from envs.env_Entropy import param
from utils.read_grid_map import ReadGridMap # 读取真实栅格地图
from envs.env_Entropy import interface2RL
import matplotlib.pyplot as plt
from utils.draw import draw_agent

class DM_env(gym.Env):
    def __init__(self):
        super(DM_env, self).__init__()
        # 参数
        self.step_count = 0
        self.max_steps = 10000
        self.seed = param.SEED
        self.width = param.global_size_width
        self.height = param.global_size_height
        # render可视化参数
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(8,4))
        # 全局变量
        self.np_random, _ = seeding.np_random(self.seed)
        self.v = 0
        self.agent_x = 0
        self.agent_y = 0
        self.agent_yaw = 0
        self.goal_x = None
        self.goal_y = None
        self.goal_yaw = None
        self.timestep = 0
        self.global_map = np.zeros((self.height, self.width), dtype=np.uint8)
        self.local_m = np.ones((param.local_size_height, param.local_size_width), dtype=np.uint8)
        self.local_m_uncertainty = np.ones((param.local_size_height, param.local_size_width), dtype=np.uint8)
        
        self.path = 0                           # buggggggggggggggggg
        self.global_occupy_map = 0              # buggggggggggggggggg
        self.global_uncertainty_map = 0         # buggggggggggggggggg

        # 定义动作空间：
        '''
        throttle_cmd, steer_cmd(当前转角)
        '''
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            shape=(2,),
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
                shape=(6,),   # [x, y, sin(yaw), cos(yaw), g_distance, g_angle]
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
        self.goal_x,self.goal_y,self.goal_yaw = self.get_random_free_position()
        self.agent_x,self.agent_y,self.agent_yaw = self.get_random_free_position()  

        # 初始化创建接口对象
        self.interface = interface2RL(self.global_map, 
                                      [self.agent_x, self.agent_y, self.agent_yaw])
        
        # 获取初始观测值（map_uncertainty、map_occupancy）
        local_m, local_m_uncertainty, _ = self.interface.ToSAC_reset()

        self.local_m = local_m
        self.local_m_uncertainty = local_m_uncertainty

        goal_distance,goal_angle = self.get_goal_dist_and_angle()
        
        # 写入观测值
        observation = {
            "map": np.stack([local_m, local_m_uncertainty], axis=0).astype(np.float32),

            "pose": np.array([
                self.agent_x / self.width,
                self.agent_y / self.height,
                math.sin(self.agent_yaw),
                math.cos(self.agent_yaw),
                goal_distance / param.max_dist,
                goal_angle / math.pi
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
        # yaw_diff = self.agent_yaw
        # yaw_diff = min(yaw_diff, 2*math.pi - yaw_diff)

        # 不确定性增益（两种可选，后续可优化这部分）
        uncertain_gain = np.mean(self.local_m_uncertainty)
        # uncertain_gain = np.max(self.local_m_uncertainty)

        # 计算 action - sub goal
        accel = param.RATIO_throttle * action[0]
        steer = param.RATIO_yaw * action[1]
        self.v += param.dt * accel
        self.v = np.clip(self.v, -10.0, 10.0)

        # 更新智能体坐标

        desired_yaw = self.v / param.L * np.tan(steer) * param.dt + self.agent_yaw
        desired_x = self.v * np.cos(self.agent_yaw) * param.dt + self.agent_x
        desired_y = self.v * np.sin(self.agent_yaw) * param.dt + self.agent_y

        sub_x = np.clip(desired_x, 0, param.global_size_width)
        sub_y = np.clip(desired_y, 0, param.global_size_height)
        sub_yaw = desired_yaw
        sub_goal = [sub_x, sub_y, sub_yaw]
        
        # 更新局部map与pose
        local_m, local_m_uncertainty, collision, current_pose = self.interface.ToSAC_step(sub_goal)

        self.agent_x = current_pose[0]
        self.agent_y = current_pose[1]
        self.agent_yaw = current_pose[2]
        self.local_m = local_m
        self.local_m_uncertainty = local_m_uncertainty
        
        goal_distance,goal_angle = self.get_goal_dist_and_angle()

        # -----填充observation-----
        observation = {
                    "map": np.stack([local_m, local_m_uncertainty], axis=0).astype(np.float32),

                    "pose": np.array([
                        self.agent_x / self.width,
                        self.agent_y / self.height,
                        math.sin(self.agent_yaw),
                        math.cos(self.agent_yaw),
                        goal_distance / param.max_dist,
                        goal_angle / math.pi
                    ], dtype=np.float32)
                }
        
        # -----reward-----
        # 计算当前时刻与终点的distance和yaw
        yaw2goal = math.atan2(self.goal_y - self.agent_y, self.goal_x - self.agent_x)
        yaw_err = abs((yaw2goal - self.agent_yaw + math.pi) % (2 * math.pi) - math.pi)
        distance_MH_new = abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)
        distance_Euclidean_new = math.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        # yaw_diff_new = self.agent_yaw
        # yaw_diff_new = min(yaw_diff_new, 2*math.pi - yaw_diff_new)

        # 不确定性增益（两种可选，后续可优化这部分）
        uncertain_gain_new = np.mean(self.local_m_uncertainty)
        # uncertain_gain = np.max(self.local_m_uncertainty)
        uncertain_gain_change = uncertain_gain_new - uncertain_gain

        reach = self.reach_goal()

        # reward计算
        distance_reward = param.DISTANCE_WEIGHT * (distance_Euclidean - distance_Euclidean_new)
        # yaw_reward = -param.YAW_WEIGHT * yaw_err
        collision_penalty = param.COLLISION_PENALTY if collision else 0.0
        step_penalty = -param.STEP_PENALTY
        reach_goal_reward = param.REACH_GOAL_REWARD if reach else 0.0
        explore_reward = -param.EXPLORE_GAIN * uncertain_gain_change
        reverse_penalty = param.REVERSE * accel if accel<0 else 0.0
        forward_reward = param.FOWARD * accel if accel>0 else 0.0
        heading_reward = param.HEADING_REWARD * (1.0 - abs(goal_angle) / math.pi)

        # 防止程序崩溃
        # 1. 防止距离坐标溢出
        distance_Euclidean = float(np.nan_to_num(distance_Euclidean, nan=500.0, posinf=500.0, neginf=500.0))
        distance_Euclidean_new = float(np.nan_to_num(distance_Euclidean_new, nan=500.0, posinf=500.0, neginf=500.0))

        # 2. 不确定性保证在 [0,1]
        uncertain_gain = float(np.nan_to_num(uncertain_gain, nan=1.0, posinf=1.0, neginf=1.0))
        uncertain_gain = max(0.0, min(1.0, uncertain_gain))

        # 3. 航向差限制在合法范围
        # yaw_diff = float(np.nan_to_num(yaw_diff, nan=0.0))
        # yaw_diff_new = float(np.nan_to_num(yaw_diff_new, nan=0.0))

        reward = (distance_reward + 
                  collision_penalty + 
                  step_penalty + 
                  reach_goal_reward + 
                  explore_reward + 
                  heading_reward +
                  reverse_penalty + 
                  forward_reward)

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
        self.ax[0].clear()
        self.ax[1].clear()

        x_min = self.agent_x - (param.local_size_width)/2
        x_max = self.agent_x + (param.local_size_width)/2
        y_min = self.agent_y - (param.local_size_height)/2
        y_max = self.agent_y + (param.local_size_height)/2

         # --- 左图：local map ---
        self.ax[0].clear()
        self.ax[0].imshow(self.local_m, cmap="gray_r", origin="lower",
                          extent=[ x_min, x_max, y_min, y_max])
        self.ax[0].set_title("Local Map")

        # --- 右图：uncertainty map ---
        self.ax[1].clear()
        self.ax[1].imshow(self.local_m_uncertainty, cmap="turbo", vmin=0, vmax=1,origin="lower",
                          extent=[x_min, x_max, y_min, y_max], # interpolation="bilinear", 
                          interpolation="bicubic")
        self.ax[1].set_title("Uncertainty Map")

        # draw agent
        draw_agent(self.ax[0], self.agent_x, self.agent_y, self.agent_yaw)
        draw_agent(self.ax[1], self.agent_x, self.agent_y, self.agent_yaw)

        plt.pause(0.2)    

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
        生成 RL 可学习的随机出生点：保证不贴墙、车头朝向大致目标、距离合适
        '''
        min_dist_to_obs = 5
        min_goal_dist = 40
        max_goal_dist = 180
        free_cells = np.argwhere(self.global_map == 0)
        max_attempts = 300
        for _ in range(max_attempts):
            # ① 随机取一个 free cell（确保不是边界）
            y, x = free_cells[self.np_random.integers(0, len(free_cells))]

            # 边界过滤
            if x < 5 or y < 5 or x > self.width - 5 or y > self.height - 5:
                continue

            # ② 保证离障碍物有安全距离
            # 使用局部窗口检查障碍
            xmin = max(0, x - min_dist_to_obs)
            xmax = min(self.width, x + min_dist_to_obs)
            ymin = max(0, y - min_dist_to_obs)
            ymax = min(self.height, y + min_dist_to_obs)

            if np.any(self.global_map[ymin:ymax, xmin:xmax] == 1):
                continue

            # ③ 如果这是 goal 的生成逻辑，需要只返回位置
            if self.goal_x is None:
                yaw = 0.0
                return x, y, yaw

            # ④ 保证起点与目标距离合理（防止前 1 步就结束）
            dx = self.goal_x - x
            dy = self.goal_y - y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_goal_dist or dist > max_goal_dist:
                continue

            # ⑤ Yaw 朝向目标 ±30°
            yaw2goal = math.atan2(dy, dx)
            yaw = yaw2goal + self.np_random.uniform(-math.pi/6, math.pi/6)

            return x, y, yaw

        # 如果实在找不到，就返回一个最安全的点
        return free_cells[0][1], free_cells[0][0], 0.0

    def reach_goal(self):
        distance = math.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        yaw_diff = abs(self.agent_yaw - self.goal_yaw)
        yaw_diff = min(yaw_diff, 2*math.pi - yaw_diff)

        if distance < 10.0 and yaw_diff < (30.0 * math.pi / 180.0):
            return True
        else:
            return False

    def get_goal_dist_and_angle(self):
        dx = self.goal_x - self.agent_x
        dy = self.goal_y - self.agent_y
        goal_distance = math.sqrt(dx*dx + dy*dy)
        goal_angle = math.atan2(dy, dx) - self.agent_yaw
        # 归一化到 [-pi, pi]
        goal_angle = (goal_angle + math.pi) % (2 * math.pi) - math.pi
        
        return goal_distance,goal_angle

    def update_global_occupy_map():
        pass

    def update_global_uncertainty_map():
        pass

if __name__ == "__main__":
    '''
    待完善部分
    1.动作空间是否可以直接设计为 throttle，yaw
    2.地图：如何随机的生成合理（具备可行性，满足越野场景）的地图
    3.奖励函数目前不完善，可能有错误
    4.状态空间是否可以加入一些视觉的东西
    5.是否要用开源数据集去跑
    6.如何用render进行可视化
        ok，已完成
    7.后续是否要用pybullet来跑
        应该是不用了，采用carla或者gazebo
    8.应该加一个倒车惩罚
    '''
    env = DM_env()   # 假如你环境叫这个名字
    obs, info = env.reset()

    for _ in range(200):
        a = env.action_space.sample()
        obs, r, d, t, info = env.step(a)
        env.render()
   

