import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # SAC_DS/
PARENT_DIR = os.path.dirname(CURRENT_DIR)                 # sb3_SAC/
sys.path.append(PARENT_DIR)

import gymnasium as gym
import scipy.ndimage as nd
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import deque
from envs.env_DS import Lidar
from envs.env_DS import interface2RL
from envs.attachments import param
from envs.attachments import Render
from envs.attachments import Planner
from envs.attachments import Frontier
from utils.read_grid_map import ReadGridMap # 读取真实栅格地图
from gymnasium import spaces
from gymnasium.utils import seeding



class DM_env(gym.Env):
    def __init__(self):
        super(DM_env, self).__init__()

        self.planner = Planner()
        self.render_obj = Render()

        # 参数
        self.micro_step = 0
        self.step_count = 0
        self.timestep = 0
        self.max_steps = 300
        self.replan_interval = 3                                      
        self.visit_count = defaultdict(int)
        self.seed = param.SEED
        self.np_random, _ = seeding.np_random(self.seed)
        self.global_map = np.zeros((param.global_size_height, param.global_size_width))
        self.reward_step = 0.0

        # vehicle state
        self.v = 0
        self.agent_x = 0
        self.agent_y = 0
        self.agent_yaw = 0
        self.goal_x = None
        self.goal_y = None
        self.goal_yaw = None
        self.global_mask = np.zeros((param.global_size_height, param.global_size_width))
        self.local_mask = np.zeros((param.local_size_height, param.local_size_width))
        # self.current_sub_goal_astar = (None, None)              # astar goal point
        self.path = deque()                                     # path without yaw

        # map old
        self.local_m = np.ones((param.local_size_height, param.local_size_width))
        self.local_m_uncertainty = np.ones((param.local_size_height, param.local_size_width))
        
        # map
        self.local_m_occ = np.zeros((param.local_size_height, param.local_size_width))              # DS证据理论
        self.local_m_free = np.zeros((param.local_size_height, param.local_size_width))             # DS证据理论
        self.local_m_unk = np.ones((param.local_size_height, param.local_size_width))               # DS证据理论
        self.m_occ = np.zeros((param.global_size_height, param.global_size_width))                  # DS证据理论
        self.m_free = np.zeros((param.global_size_height, param.global_size_width))                 # DS证据理论
        self.m_unk =np.zeros((param.global_size_height, param.global_size_width))                   # DS证据理论
        self.belief_map = defaultdict(lambda: {"occ": 0.0, "free": 0.0, "unk": 1.0})

        # 评价指标
        self.episode_reward = 0.0
        self.episode_length = 0

        # 状态空间通道数
        self.global_map_channel = 4         
        self.local_map_channel = 4         
        self.pose_channel = 7              

        # 动作空间通道数
        self.action_channel = 3             

        # 定义状态空间：
        '''
        global_mask
        m_occ
        m_free
        m_unk
        local_mask
        local_m_occ
        local_m_free
        local_m_unk
        x, y, sin(yaw), cos(yaw), g_distance, g_angle, v_norm(归一化到 [-1, 1])
        '''
        self.observation_space = spaces.Dict({
            "global_map": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.global_map_channel, 
                       param.global_size_height, 
                       param.global_size_width),  
                dtype=np.float32
            ),

            "local_map": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.local_map_channel, 
                       param.local_size_height, 
                       param.local_size_width),  
                dtype=np.float32
            ),

            "pose": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.pose_channel,), 
                dtype=np.float32
            )
        })

        # 定义动作空间：
        '''
        alpha_size          frontier score函数 size权重
        alpha_dist          frontier score函数 distance权重
        alpha_risk          frontier score函数 risk权重
        '''
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
            shape=(self.action_channel,),
            dtype=np.float32
        ) 

    def reset(self, *, seed=None, options=None):
        '''
        -重置环境状态
        -返回值
        observation	    初始观察
        info	        额外信息（一般空字典）
        '''
        super().reset(seed=seed)

        self.episode_reward = 0.0
        self.episode_length = 0
        self.step_count = 0
        self.micro_step = 0
        self.visit_count.clear()
        self.v = 0.0
        self.goal_x = None
        self.goal_y = None
        self.goal_yaw = None
        self.path = deque()

        # 生成地图
        # self.generate_random_map(seed = seed)     # 随机生成地图 
        self.get_easy_map()                         # 生成简单测试地图

        # 随机生成智能体和目标点位置
        self.goal_x,self.goal_y,self.goal_yaw = self.get_random_free_position()
        self.agent_x,self.agent_y,self.agent_yaw = self.get_random_free_position()  

        # 得到 position mask
        self.global_mask[self.agent_y // param.XY_RESO, self.agent_x // param.XY_RESO] = 1
        self.local_mask[param.local_size_height//2, param.local_size_width//2] = 1

        # 初始化创建接口对象
        self.interface = interface2RL(self.global_map, 
                                      [self.agent_x, self.agent_y, self.agent_yaw])

        # 获取初始观测值（map_uncertainty、map_occupancy）
        m_occ, m_free, m_unk, local_m, local_m_occ, local_m_free, local_m_unk, _ , self.belief_map = self.interface.ToSAC_reset()

        self.local_m_occ = local_m_occ
        self.local_m_free = local_m_free
        self.local_m_unk = local_m_unk
        self.local_m = local_m
        self.m_occ = m_occ
        self.m_free = m_free
        self.m_unk = m_unk

        goal_distance = self.get_distance2goal()
        goal_angle = self.get_angle2goal()
        
        # 写入观测值
        observation = {
            "global_map": np.stack([self.global_mask,
                                    self.m_occ,
                                    self.m_free,
                                    self.m_unk], axis=0).astype(np.float32),

            "local_map": np.stack([self.local_mask,
                                   self.local_m_occ,
                                   self.local_m_free,
                                   self.local_m_unk], axis=0).astype(np.float32),

            "pose": np.array([
                self.agent_x / param.global_size_width,
                self.agent_y / param.global_size_height,
                math.sin(self.agent_yaw),
                math.cos(self.agent_yaw),
                goal_distance / param.max_dist,
                goal_angle / math.pi,
                self.v / param.max_velocity
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
        self.timestep += 1
        self.reward_step = 0.0

        

        # ---计算 frontier , astar planning---
        self.planning_strategy(action)

        # micro steps
        for i in range(self.replan_interval):
            self.micro_step += 1

            # pop sub_goal
            '''
            buggggggggggggggg:sub_goal maybe None
            '''
            sub_goal = self.pop_sub_goal()
            if sub_goal is None:
                self.planning_strategy(action)
                sub_goal = self.pop_sub_goal()

            # last reward params
            distance, angle_yaw_goal, uncertain_gain, risk = self.calculate_reward_param()

            # ---更新局部map与pose---
            m_occ, m_free, m_unk, local_m, local_m_occ, local_m_free, local_m_unk, collision, current_pose, self.belief_map = self.interface.ToSAC_step(sub_goal)

            self.agent_x = current_pose[0]
            self.agent_y = current_pose[1]
            self.agent_yaw = current_pose[2]
            self.local_m_occ = local_m_occ
            self.local_m_free = local_m_free
            self.local_m_unk = local_m_unk
            self.local_m = local_m
            self.m_occ = m_occ
            self.m_free = m_free
            self.m_unk = m_unk

            # 得到 position mask
            self.global_mask[self.agent_y // param.XY_RESO, self.agent_x // param.XY_RESO] = 1
            
            goal_distance = self.get_distance2goal()
            goal_angle = self.get_angle2goal()

            # ---reward---
            # reward params
            distance_new, angle_yaw_goal_new, uncertain_gain_new, risk_new = self.calculate_reward_param()
            dist_err = distance_new - distance
            yaw2goal_err = abs(angle_yaw_goal_new) - abs(angle_yaw_goal) # 
            uncertainty_err = uncertain_gain_new - uncertain_gain
            risk_err = risk_new - risk
            reach = self.reach_goal()
            visit_penalty = self.revisit_penalty_func()

            # reward计算
            distance_reward = param.DISTANCE_WEIGHT * (-dist_err)
            yaw_reward = param.YAW_WEIGHT * (-yaw2goal_err)
            collision_penalty = param.COLLISION_WEIGHT if collision else 0.0
            step_penalty = param.STEP_PENALTY_WEIGHT * (-self.micro_step / self.max_steps)
            reach_goal_reward = param.REACH_GOAL_WEIGHT if reach else 0.0
            explore_reward = param.EXPLORE_GAIN_WEIGHT * (-uncertainty_err)
            reverse_penalty = param.REVERSE_WEIGHT * self.v if self.v<0 else 0.0
            forward_reward = param.FOWARD_WEIGHT * self.v if self.v>0 else 0.0
            risk_penalty = param.RISK_PENALTY_WEIGHT * (-risk_err)
            revisit_penalty = param.VISIT_PENALTY_WEIGHT * (-visit_penalty)

            self.reward_step += (distance_reward + 
                                 yaw_reward + 
                                 collision_penalty + 
                                 step_penalty + 
                                 reach_goal_reward + 
                                 explore_reward +
                                 reverse_penalty + 
                                 forward_reward + 
                                 risk_penalty +
                                 revisit_penalty)

            #---terminated collsion reach---
            if (reach or collision): 
                break
            if self.step_count >= self.max_steps:
                break
    
        self.reward_step = np.clip(self.reward_step, -200, 200)

        # ---observation---
        obs = {
                    "global_map": np.stack([self.global_mask,
                                            self.m_occ,
                                            self.m_free,
                                            self.m_unk], axis=0).astype(np.float32),

                    "local_map": np.stack([self.local_mask,
                                           self.local_m_occ,
                                           self.local_m_free,
                                           self.local_m_unk], axis=0).astype(np.float32),

                    "pose": np.array([
                        self.agent_x / param.global_size_width,
                        self.agent_y / param.global_size_height,
                        math.sin(self.agent_yaw),
                        math.cos(self.agent_yaw),
                        goal_distance / param.max_dist,
                        goal_angle / math.pi,
                        self.v / param.max_velocity
                    ], dtype=np.float32)
                }

        # ---评价指标---
        self.episode_reward += self.reward_step
        self.episode_length += 1

        # ---输出info---
        info = {
            "reach_goal": reach,
            "collision": collision,
            "distance_to_goal": distance_new,
            "uncertainty": np.mean(local_m_unk)
        }

        # ---设置terminated和truncated---
        terminated = True if (reach or collision) else False
        truncated = True if self.step_count >= self.max_steps else False

        # 如果 episode 结束，必须手动写入 episode 信息
        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,   # 总奖励
                "l": self.episode_length    # episode 长度
            }

        # ---调试信息---
        if reach:
            print("reachhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")

        if collision:
            print("collisionnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        
        if truncated:
            print("max   steppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp")
        
        print("current.pose:",self.agent_x,self.agent_y)
        print("goal:",self.goal_x,self.goal_y)
        print("reward:",self.reward_step)
        print("step:",self.timestep)
        
        return obs, self.reward_step, terminated, truncated, info

    def get_frontier_clusters(self, action): 
        '''
        获取 top k frontier clusters
        '''
        frontier = Frontier(self.local_m_occ, self.local_m_unk, 
                            self.local_m_free,(param.local_size_height // 2, param.local_size_width // 2)
                            , action, k=param.frontier_k)
        frontier_mask = frontier.compute_frontier_mask()
        clusters = frontier.cluster_frontiers(frontier_mask)
        infos = frontier.summarize_clusters_rep_points(clusters)
        topk = frontier.select_topk(infos) # no risk map for now
        return topk, frontier_mask

    def render(self, mode='human'):
        '''
        渲染环境（可选）
        mode: 渲染模式
        'human'：渲染到屏幕
        'rgb_array'：返回RGB图像数组
        '''
        
        self.render_obj._draw_local_map(self.local_m, self.agent_x, self.agent_y, self.agent_yaw)
        self.render_obj._draw_local_map_uncertainty(self.local_m_unk, self.agent_x, self.agent_y, self.agent_yaw)
        self.render_obj._draw_belief_map(self.belief_map)
        self.render_obj.flush()
    
        plt.pause(0.10)   

    def planning_strategy(self, action):
        topk, _ = self.get_frontier_clusters(action)
        best = topk[0] if topk and topk[0] is not None else None
        current_sub_goal_astar = best.get("rep_ij") if best else None     
            # 
        self.path.clear()
        if current_sub_goal_astar is None:
            print("No frontier found!")
            current_sub_goal_ = self.sub_goal_back_up_poloicy()
            path_local = self.get_path_astar(self.local_m, current_sub_goal_)
            path_global = self.path_local_to_global(path_local, self.agent_x, self.agent_y)
            self.path.extend(path_global)
        else:
            path_local = self.get_path_astar(self.local_m, current_sub_goal_astar)
            path_global = self.path_local_to_global(path_local, self.agent_x, self.agent_y)
            self.path.extend(path_global)


    def pop_sub_goal(self):
        '''
        返回 sub_goal = [x, y, yaw]
        如果 path 不足，返回 None（触发重规划或 fallback）
        '''
        if len(self.path) == 0:
            return None
        
        if len(self.path) == 1:
            p_ = self.path.popleft()
            sub_x, sub_y = p_
            sub_yaw = self.agent_yaw
            sub_goal = [sub_x, sub_y, sub_yaw]
            return sub_goal
        else:
            p_ = self.path.popleft()
            p_next = self.path[0]
            sub_x, sub_y = p_
            sub_x_next, sub_y_next = p_next
            sub_yaw = math.atan2((sub_y_next - sub_y),(sub_x_next - sub_x))
            sub_goal = [sub_x, sub_y, sub_yaw]
            return sub_goal

    def sub_goal_back_up_poloicy(self):
        H, W = param.local_size_height, param.local_size_width
        ci, cj = H // 2, W // 2
        res = param.XY_RESO

        # goal 在 local 坐标系中的偏移（格）
        dx = (self.goal_x - self.agent_x) / res
        dy = (self.goal_y - self.agent_y) / res

        # local 索引 (y, x)
        y = int(round(ci + dy))
        x = int(round(cj + dx))

        # clamp 到 local 边界
        y = int(np.clip(y, 0, H - 1))
        x = int(np.clip(x, 0, W - 1))
        return (x, y)


    def path_local_to_global(self, path_local, agent_x, agent_y):
        if path_local is None or len(path_local)==0:
            return []
        H, W = param.local_size_height, param.local_size_width
        ci, cj = H // 2, W // 2
        res = param.XY_RESO
        
        path_global = []
        for (y, x) in path_local:   # y, x
            xg = agent_x + (x - cj) * res
            yg = agent_y + (y - ci) * res
            # 
            xg = np.clip(xg, 0, param.global_size_width - 1)
            yg = np.clip(yg, 0, param.global_size_height - 1)
            path_global.append((xg, yg))

        return path_global

    # Astar interface
    def get_path_astar(self, local_map, goal):
        path = self.planner.main_workflow(local_map, goal)
        return path

    def calculate_reward_param(self):

        # distance and yaw
        distance_MH = abs(self.agent_x - self.goal_x) + abs(self.agent_y - self.goal_y)
        distance_Euclidean = math.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        yaw2goal = self.get_angle2goal()
        angle_yaw_goal = yaw2goal - self.agent_yaw
        angle_yaw_goal = (angle_yaw_goal + math.pi) % (2 * math.pi) - math.pi

        # 不确定性增益
        uncertain_gain = np.mean(self.local_m_unk)
        # uncertain_gain = np.max(self.local_m_unk)

        # risk gain
        ci = param.local_size_height // 2
        cj = param.local_size_width // 2
        risk_gain = self.local_m_occ[ci, cj] + self.local_m_unk[ci, cj] 
        
        return distance_Euclidean, angle_yaw_goal, uncertain_gain, risk_gain
        
    def revisit_penalty_func(self):
        ix, iy = int(self.agent_x), int(self.agent_y)
        c = self.visit_count[(ix, iy)]
        visit_penalty = c 
        self.visit_count[(ix, iy)] = c + 1
        return visit_penalty

    def generate_random_map(self, obstacle_ratio=0.2, seed = None): # bugggggggggggggggggggggggggggggggggggg
        """
        使用 gym 环境的随机系统生成随机地图
        1 = obstacle
        0 = free
        """
        self.np_random, _ = seeding.np_random(seed)

        random_matrix = self.np_random.random((param.global_size_height, param.global_size_width))
        self.global_map = (random_matrix < obstacle_ratio).astype(np.float32)

    def get_easy_map(self, width=256, height=256):
        map_ = np.zeros((height, width))

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
            # 随机取一个 free cell（确保不是边界）
            y, x = free_cells[self.np_random.integers(0, len(free_cells))]

            # 边界过滤
            if x < 5 or y < 5 or x > param.global_size_width - 5 or y > param.global_size_height - 5:
                continue

            # 保证离障碍物有安全距离
            # 使用局部窗口检查障碍
            xmin = max(0, x - min_dist_to_obs)
            xmax = min(param.global_size_width, x + min_dist_to_obs)
            ymin = max(0, y - min_dist_to_obs)
            ymax = min(param.global_size_height, y + min_dist_to_obs)

            if np.any(self.global_map[ymin:ymax, xmin:xmax] == 1):
                continue

            # 如果这是 goal 的生成逻辑，需要只返回位置
            if self.goal_x is None:
                yaw = 0.0
                return x, y, yaw

            # 保证起点与目标距离合理
            dx = self.goal_x - x
            dy = self.goal_y - y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_goal_dist or dist > max_goal_dist:
                continue

            # Yaw 朝向目标 ±30°
            yaw2goal = math.atan2(dy, dx)
            yaw = yaw2goal + self.np_random.uniform(-math.pi/6, math.pi/6)

            return x, y, yaw

        # 如果实在找不到，就返回一个最安全的点
        return free_cells[0][1], free_cells[0][0], 0.0

    def reach_goal(self):
        distance = math.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        yaw_diff = abs(self.agent_yaw - self.goal_yaw)
        yaw_diff = min(yaw_diff, 2*math.pi - yaw_diff)

        if distance < 5.0: # and yaw_diff < (30.0 * math.pi / 180.0):
            return True
        else:
            return False

    def get_distance2goal(self):
        dx = self.goal_x - self.agent_x
        dy = self.goal_y - self.agent_y
        goal_distance = math.sqrt(dx*dx + dy*dy)
        
        return goal_distance
    
    def get_angle2goal(self):
        goal_angle = math.atan2(self.goal_y - self.agent_y, self.goal_x - self.agent_x)
        # 归一化到 [-pi, pi]
        goal_angle = (goal_angle + math.pi) % (2 * math.pi) - math.pi
        
        return goal_angle

if __name__ == "__main__":
    '''
    
    '''
    pass
   

