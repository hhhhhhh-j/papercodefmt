import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import time
from loguru import logger
from collections import defaultdict
from utils.read_grid_map import ReadGridMap
from interactions.attachments import Frontier
from interactions.attachments import param


class interface2RL:
    def __init__(self, global_map, init_pose = [0.0, 0.0, 0.0]):
        # 创建调用对象
        self.global_map = global_map
        self.lidar = Lidar(global_map)
        
        # 全局变量
        # self.local_m = np.zeros((param.local_size_height, param.local_size_width))
        # self.local_m_uncertainty = np.zeros((param.local_size_height, param.local_size_width))
        self.current_pose = np.array(init_pose, dtype=np.float32) 

        self.belief_map_dict = defaultdict(lambda: {"occ": 0.0, "free": 0.0, "unk": 1.0})
        
        self.local_m_occ = np.zeros((param.local_size_height, param.local_size_width))              # DS证据理论
        self.local_m_free = np.zeros((param.local_size_height, param.local_size_width))             # DS证据理论
        self.local_m_unk = np.ones((param.local_size_height, param.local_size_width))               # DS证据理论
        self.m_occ = np.zeros((param.global_size_height, param.global_size_width), dtype=float)     # DS证据理论
        self.m_free = np.zeros((param.global_size_height, param.global_size_width), dtype=float)    # DS证据理论
        self.m_unk =np.zeros((param.global_size_height, param.global_size_width), dtype=float)      # DS证据理论

    def ToSAC_reset(self):
        '''
        与RL reset的接口
        '''
        res = self.lidar.DS_update([self.current_pose[0], 
                         self.current_pose[1], 
                         self.current_pose[2]])
        # 防止map无效
        if res is None:
            self.m_occ = np.zeros((param.global_size_height, param.global_size_width))
            self.m_free = np.zeros((param.global_size_height, param.global_size_width))
            self.m_unk = np.ones((param.global_size_height, param.global_size_width))
            self.local_m_occ = np.zeros((param.local_size_height, param.local_size_width))
            self.local_m_free = np.zeros((param.local_size_height, param.local_size_width))
            self.local_m_unk = np.ones((param.local_size_height, param.local_size_width))
        else:
            self.m_occ,self.m_free,self.m_unk = res
            
            self.local_m_occ = self.get_local_map(self.m_occ,self.current_pose ) 
            self.local_m_free = self.get_local_map(self.m_free, self.current_pose )
            self.local_m_unk = self.get_local_map(self.m_unk, self.current_pose )

        local_m = self.local_m_occ # + 0.5 * self.local_m_unk
        belief_map_dict = self.record_belief_map()
        
        return self.m_occ, self.m_free, self.m_unk, local_m, self.local_m_occ, self.local_m_free, self.local_m_unk, self.current_pose, belief_map_dict

    def ToSAC_step(self, sub_goal):
        '''
        与RL step的接口
        '''
        x_new = sub_goal[0]
        y_new = sub_goal[1]
        yaw_new = sub_goal[2]

        self.m_occ, self.m_free, self.m_unk = self.lidar.DS_update([x_new, y_new, yaw_new])
        self.current_pose = np.array([x_new, y_new, yaw_new])
        
        self.local_m_occ = self.get_local_map(self.m_occ,self.current_pose )
        self.local_m_unk = self.get_local_map(self.m_unk, self.current_pose )
        self.local_m_free = self.get_local_map(self.m_free, self.current_pose )

        # belief map
        if not hasattr(self, "_rec_step"): # 初始化记录步数
            self._rec_step = 0
        self._rec_step += 1

        if self._rec_step  == 1:   # 第1步记录一次
            self.record_belief_map()

        if self._rec_step % 10 == 0:   # 每10步记录一次
            self.record_belief_map()


        # 是否碰撞
        collision = self.is_collision(self.current_pose)

        local_m = self.local_m_occ  + 0.5 * self.local_m_unk
        
        return self.m_occ, self.m_free, self.m_unk, local_m, self.local_m_occ, self.local_m_free, self.local_m_unk, collision, self.current_pose, self.belief_map_dict

    def record_belief_map(self):
        agent_x = self.current_pose[0]
        agent_y = self.current_pose[1]

        H, W = param.local_size_height, param.local_size_width  # e.g., 64, 64
        center_i = H // 2
        center_j = W // 2

        for i in range(H):
            for j in range(W):

                # local → global 坐标偏移
                dx = j - center_j
                dy = i - center_i

                # global 坐标
                gx = int(agent_x + dx)
                gy = int(agent_y + dy)

                gx = np.clip(gx, 0, param.global_size_width - 1)
                gy = np.clip(gy, 0, param.global_size_height - 1)

                # 写入 belief_map_dict（注意自动初始化）
                self.belief_map_dict[(gx, gy)]["occ"] = float(self.local_m_occ[i, j])
                self.belief_map_dict[(gx, gy)]["free"] = float(self.local_m_free[i, j])
                self.belief_map_dict[(gx, gy)]["unk"] = float(self.local_m_unk[i, j])

        return self.belief_map_dict

    def get_local_map(self, global_map, X):
        '''
        获取局部map
        '''
        x = int(X[0])
        y = int(X[1])

        h = param.local_size_height
        w = param.local_size_width

        half_h = h // 2
        half_w = w // 2

         # 裁剪区域的左上/右下坐标
        y_min = y - half_h
        y_max = y + half_h
        x_min = x - half_w
        x_max = x + half_w

        # 计算真实可裁剪区域（保证不越界）
        real_y_min = max(0, y_min)
        real_y_max = min(global_map.shape[0], y_max)
        real_x_min = max(0, x_min)
        real_x_max = min(global_map.shape[1], x_max)

        local_map = global_map[real_y_min:real_y_max, real_x_min:real_x_max]

         # 计算需要补多少
        pad_top = real_y_min - y_min          # y_min < 0 → 需要补上方
        pad_bottom = y_max - real_y_max       # y_max > height → 补下方
        pad_left = real_x_min - x_min         # x_min < 0 → 补左方
        pad_right = x_max - real_x_max        # x_max > width → 补右方

        # 如果大小不够，补齐（padding）
        local_map = np.pad(
            local_map,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=1   # 未知区域给 1
        )

        assert local_map.shape == (h, w), f"Local map shape incorrect: {local_map.shape}"

        return local_map

    def get_local_obs(self, local_map, threshold=0.6):
        '''
        获取局部地图中的障碍物坐标
        '''
        h, w = local_map.shape
         # 原始障碍物点（概率大于阈值）
        oy, ox = np.where(local_map > threshold)

        # 上边界
        top_x = np.arange(w)
        top_y = np.zeros(w, dtype=int)

        # 下边界
        bottom_x = np.arange(w)
        bottom_y = np.ones(w, dtype=int) * (h - 1)

        # 左边界
        left_x = np.zeros(h, dtype=int)
        left_y = np.arange(h)

        # 右边界
        right_x = np.ones(h, dtype=int) * (w - 1)
        right_y = np.arange(h)

        # 合并所有坐标
        all_x = np.concatenate([ox, top_x, bottom_x, left_x, right_x])
        all_y = np.concatenate([oy, top_y, bottom_y, left_y, right_y])
        return all_x.tolist(), all_y.tolist()
    
    def is_collision(self, pose):
        '''
        检测碰撞
        '''
        x = int(pose[0])
        y = int(pose[1])

        if x < 0 or x >= self.global_map.shape[1] or y < 0 or y >= self.global_map.shape[0]:
            return True  # 越界视为碰撞

        if self.global_map[y, x] >= 0.6:  # 假设大于等于0.6的概率视为障碍物
            return True
        else:
            return False

    '''Hybrid A* interface
    def GetPathHAstar(self, sx = 10.0, sy = 7.0, syaw0 = np.deg2rad(120.0), 
                gx = 45.0, gy = 20.0, gyaw0 = np.deg2rad(90.0)): # , sx, sy, syaw0, gx, gy, gyaw0, ox, oy
       
        # 生成障碍物的坐标
        ox, oy = self.get_local_obs(self.global_map)
    
        t0 = time.time()

        # 使用Hybrid A*算法进行路径规划
        path = planning.hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0,ox, oy, 
                                              param.XY_RESO, param.YAW_RESO)
        t1 = time.time()
        print("running T: ", t1 - t0)
    
        if not path:
            print("Searching failed!")
            return
    
        # 提取路径的信息
        x = path.x
        y = path.y
        yaw = path.yaw
        direction = path.direction

        return x,y, yaw, direction
        '''
    
class Lidar:
    def __init__(self, true_map):
        self.true_map = true_map
        self.reso = param.RESO
        # Parameters for the sensor model.
        self.meas_phi = param.MEAS_PHI
        self.rmax = param.RMAX
        self.alpha = param.ALPHA
        self.beta = param.BETA
        # lidar全局变量
        (self.M, self.N) = np.shape(true_map)
        # self.L0 = np.zeros((self.M, self.N))                # Prior log-odds
        # self.L = np.zeros((self.M, self.N))
        # self.m = np.full((self.M, self.N), 0.5)             # 初始化感知概率地图为0.5
        # self.ms = []                                        # 存储每一帧的感知概率地图
        # self.m_uncertainty = np.full((self.M, self.N),1)    # 初始化不确定性地图
        # self.m_uncertaintys = []                            # 存储每一帧的不确定性地图
        
        self.m_occ = np.zeros((self.M, self.N), dtype=float)            # DS证据理论
        self.m_free = np.zeros((self.M, self.N), dtype=float)           # DS证据理论
        self.m_unk = np.ones((self.M, self.N), dtype=float)             # DS证据理论

    def get_ranges(self, X):
        """
        true_map : 真实环境栅格 (M, N)，0=可行驶，1=障碍
        X        : 当前车辆/雷达状态 [x, y, theta]
        meas_phi : 激光束相对车体的角度数组，例如 np.arange(-0.4, 0.4, 0.05)
        rmax     : 最大量程
        返回值   : 每条激光束的量测距离数组 meas_r
        """
        x = X[0] / self.reso
        y = X[1] / self.reso
        theta = X[2]
        meas_r = self.rmax * np.ones(self.meas_phi.shape)
        
        # Iterate for each measurement bearing.
        for i in range(len(self.meas_phi)):
            # Iterate over each unit step up to and including rmax.
            for r in range(1, self.rmax+1):
                # Determine the coordinates of the cell.
                xi = int(round(x + r * math.cos(theta + self.meas_phi[i])))
                yi = int(round(y + r * math.sin(theta + self.meas_phi[i])))
                
                # If not in the map, set measurement there and stop going further.
                if (xi <= 0 or xi >= self.N-1 or yi <= 0 or yi >= self.M-1):
                    meas_r[i] = r
                    break
                # If in the map, but hitting an obstacle, set the measurement range
                # and stop ray tracing.
                elif self.true_map[int(round(yi)), int(round(xi))] == 1:
                    meas_r[i] = r
                    break
                    
        return meas_r

    def inverse_scanner_DS(self, X, meas_r):
        """
        
        """
        x_ind, y_ind, theta = X[0]/self.reso, X[1]/self.reso, X[2]

        r_grid = int(self.rmax / self.reso)

        x_min = max(0, int(x_ind - r_grid))
        x_max = min(self.N, int(x_ind + r_grid)+1)
        y_min = max(0, int(y_ind - r_grid))
        y_max = min(self.M, int(y_ind + r_grid)+1)

        # 构建 meshgrid
        xs = np.arange(x_min, x_max)
        ys = np.arange(y_min, y_max)
        yy, xx= np.meshgrid(ys, xs, indexing='ij')

        # 距离 r
        r = np.sqrt((xx - x_ind)**2 + (yy - y_ind)**2)

        # 角度 phi
        phi = (np.arctan2(yy - y_ind, xx - x_ind) - theta + np.pi) % (2 * np.pi) - np.pi

        # 初始化
        m_occ = np.full_like(r, 0.0, dtype=float)
        m_unk = np.full_like(r, 1.0, dtype=float)
        m_free = np.full_like(r, 0.0, dtype=float)

        # 对每条激光束判断
        for k, angle in enumerate(self.meas_phi):
            # 添加噪声
            '''
            distance angle noise
            '''
            w_r = np.exp(-0.01 * r)   # 距离越大，可信度越低
            w_phi = np.exp(-( (phi - angle)**2 ) / (2 * self.beta**2))
            meas_r_noisy = meas_r[k] + np.random.normal(0, 0.05)

            mask_angle = np.abs(phi - angle) <= self.beta / 2

            # mask_occ = mask_angle & (np.abs(r - meas_r[k]) <= self.alpha / 2)
            # mask_free = mask_angle & (r < meas_r[k])
            mask_occ = mask_angle & (np.abs(r - meas_r_noisy) <= self.alpha / 2)
            mask_free = mask_angle & (r < meas_r_noisy)

            w = w_r * w_phi

            # m_occ[mask_occ] += 0.7 * w[mask_occ]
            # m_unk[mask_occ] = 1 - m_occ[mask_occ]
            # m_free[mask_occ] = 0.0

            # m_free[mask_free] += 0.3 * w[mask_free]
            # m_unk[mask_free] = 1 - m_free[mask_free]
            # m_occ[mask_free] = 0.0

            # 0120 调整
            occ_evi = 0.7 * w
            free_evi = 0.3 * w

            m_occ[mask_occ] = np.maximum(m_occ[mask_occ], occ_evi[mask_occ])
            m_free[mask_free] = np.maximum(m_free[mask_free], 1 - m_occ[mask_free])

        m_occ = np.clip(m_occ, 0.0, 1.0)
        m_free = np.clip(m_free, 0.0, 1.0)

        s = m_occ + m_free
        over = s > 1.0
        if np.any(over):
            m_occ[over] /= s[over]
            m_free[over] /= s[over]
            s = m_occ + m_free
        
        m_unk = 1.0 - s
        m_unk = np.clip(m_unk, 0.0, 1.0)

        # m_occ = np.clip(m_occ, 0, 1)
        # m_free = np.clip(m_free, 0, 1)
        # m_unk = np.clip(m_unk, 0, 1)

        return m_occ, m_free, m_unk
    
    def Dempster_combination(self, m1_occ, m1_free, m1_unk, m2_occ, m2_free, m2_unk):
        '''
        Dempster_combination_rule 的 Docstring
        '''
        # 冲突
        K = m1_occ * m2_free + m1_free * m2_occ
        K = np.clip(K, 0.0, 0.999999)

        coef = 1.0 / (1.0 - K)

        # ---------  occ ----------
        m_occ = coef * (
            m1_occ * m2_occ +
            m1_occ * m2_unk +
            m1_unk * m2_occ
        )

        # ---------  free ----------
        m_free = coef * (
            m1_free * m2_free +
            m1_free * m2_unk +
            m1_unk * m2_free
        )

        # --------- unknown ----------
        m_unk = coef * (m1_unk * m2_unk)

        return m_occ, m_free, m_unk

    def DS_update(self, X):
        '''
        DS_update 
        '''
        x_ind, y_ind = X[0]/self.reso, X[1]/self.reso

        r_grid = int(self.rmax / self.reso)

        x_min = max(0, int(x_ind - r_grid))
        x_max = min(self.N, int(x_ind + r_grid)+1)
        y_min = max(0, int(y_ind - r_grid))
        y_max = min(self.M, int(y_ind + r_grid)+1)

        meas_r = self.get_ranges(X)
        # 从全局地图中取旧证据
        m1_occ = self.m_occ[y_min:y_max, x_min:x_max]
        m1_free = self.m_free[y_min:y_max, x_min:x_max]
        m1_unk  = self.m_unk[y_min:y_max, x_min:x_max]
        m2_occ,m2_free,m2_unk = self.inverse_scanner_DS(X,meas_r)

        m_occ_new, m_free_new, m_unk_new = self.Dempster_combination(m1_occ, m1_free, m1_unk, m2_occ, m2_free, m2_unk)

        self.m_occ[y_min:y_max, x_min:x_max] = m_occ_new
        self.m_free[y_min:y_max, x_min:x_max] = m_free_new
        self.m_unk[y_min:y_max, x_min:x_max]  = m_unk_new

        # 全局轻微遗忘（可选，防止“记死”）
        self.fading(param.gamma)

        return self.m_occ,self.m_free,self.m_unk



    def fading(self, gamma = 0.98):
        '''
        fading 的 Docstring
        '''
        self.m_occ *= gamma
        self.m_free *= gamma
        self.m_unk = 1 - self.m_occ - self.m_free
        
        self.m_occ = np.clip(self.m_occ, 0.0, 1.0)
        self.m_free = np.clip(self.m_free, 0.0, 1.0)
        self.m_unk = np.clip(self.m_unk, 0.0, 1.0)
    
    # def get_uncertainty_map_DS(self):
    #     """
    #     生成不确定性地图
    #     """
    #     self.m_uncertainty = -(self.m * np.log(self.m) + (1 - self.m) * np.log(1 - self.m))
    #     self.m_uncertainty /= np.log(2)
    #     self.m_uncertainty = np.clip(self.m_uncertainty, 1e-6, 1 - 1e-6)

    #     self.m_uncertaintys.append(self.m_uncertainty)

    #     return self.m_uncertainty
    
    # def get_uncertainty_map_DS():


    # def generate_probability_map(self, X):
    #     """
    #     生成感知概率地图
    #     """
    #     meas_rs = []
    #     invmods = []
        
    #     meas_r = self.get_ranges(X)
    #     meas_rs.append(meas_r)
    #     invmod = self.inverse_scanner(X,meas_r)
    #     invmod = np.clip(invmod, 0.1, 0.99)  # 避免log(0)
    #     invmods.append(invmod)
        
    #     # 计算逆模型对应的 log-odds
    #     inv_logodds = np.log(invmod / (1 - invmod))
    #     # 更新
    #     self.L = self.L + inv_logodds - self.L0
    #     # Clamp 避免爆炸
    #     self.L = np.clip(self.L, -20, 20)

    #     # Calculate a grid of probabilities from the log odds.
    #     self.m = np.divide(np.exp(self.L), np.add(1, np.exp(self.L)))
    #     self.m = np.clip(self.m, 1e-6, 1 - 1e-6)

    #     self.ms.append(self.m)

    #     return self.m

    def update(self, X):
        """
        更新感知概率地图和不确定性地图
        """
        X_clipped = np.array([
            np.clip(X[0], 0, self.N * self.reso - 1),
            np.clip(X[1], 0, self.M * self.reso - 1),
            X[2]   
        ])
        m = self.generate_probability_map(X_clipped)
        m_uncertainty = self.get_uncertainty_map_entropy()

        return m, m_uncertainty

if __name__ == "__main__":
    pass
