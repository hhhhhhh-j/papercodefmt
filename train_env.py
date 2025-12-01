import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from read_grid_map import ReadGridMap
import time

class param:
    # Hybrid A* 参数
    XY_RESO = 1.0                   # 网格地图分辨率
    YAW_RESO = np.deg2rad(15.0)     # 航向角分辨率
    WB = 3.5                        # 车辆的轴距 (WheelBase)
    MOVE_STEP = 0.4                 # 插值分辨率，表示在路径中每个节点之间的距离
    # 地图尺寸
    local_size_width = 64
    local_size_height = 64
    global_size_width = 256
    global_size_height = 256
    SEED = None                     # 随机地图种子
    # action 缩放系数
    RATIO_x = 5               # x缩放系数
    RATIO_y = 5               # y缩放系数
    RATIO_yaw = math.pi / 4      # yaw缩放系数
    # reward 系数
    REACH_GOAL_REWARD = 100.0       # 到达目标奖励
    COLLISION_PENALTY = -100.0      # 碰撞惩罚
    STEP_PENALTY = 0.0              # 每步惩罚
    DISTANCE_WEIGHT = 2.0           # 距离权重
    YAW_WEIGHT = -0.5               # 航向角权重
    EXPLORE_GAIN = 10.0             # 探索奖励增益
    MOVE_REWARD = 10000.0           # 防止局部塌缩
    # lidar参数
    RESO = 1
    MEAS_PHI = np.arange(-0.4, 0.4, 0.05)
    RMAX = 30 # Max beam range.
    ALPHA = 1 # Width of an obstacle (distance about measurement to fill in).
    BETA = 0.05 # Angular width of a beam.

class interface2RL:
    def __init__(self, global_map, init_pose = [0.0, 0.0, 0.0]):
        # 创建调用对象
        self.global_map = global_map
        self.lidar = Lidar(global_map)
        # 全局变量
        self.local_m = np.zeros((param.local_size_height, param.local_size_width))
        self.local_m_uncertainty = np.zeros((param.local_size_height, param.local_size_width))
        self.current_pose = np.array(init_pose, dtype=np.float32) 

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

        y_min = max(0, y - half_h)
        y_max = min(global_map.shape[0], y + half_h)

        x_min = max(0, x - half_w)
        x_max = min(global_map.shape[1], x + half_w)

        local_map = global_map[y_min:y_max, x_min:x_max]

        # 如果大小不够，补齐（padding）
        local_map = np.pad(
            local_map,
            ((0, h - local_map.shape[0]), (0, w - local_map.shape[1])),
            mode='constant',
            constant_values=1   # 未知区域给 1
        )
        
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

    def GetPath(self, sx = 10.0, sy = 7.0, syaw0 = np.deg2rad(120.0), 
                gx = 45.0, gy = 20.0, gyaw0 = np.deg2rad(90.0)): # , sx, sy, syaw0, gx, gy, gyaw0, ox, oy
        '''
        获取行动路径
        '''
       
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

    def ToSAC_reset(self):
        '''
        与RL reset的接口
        '''
        res = self.lidar.update([self.current_pose[0], 
                         self.current_pose[1], 
                         self.current_pose[2]])
        # 防止map无效
        if res is None:
            self.local_m = np.ones((param.local_size_height, param.local_size_width))
            self.local_m_uncertainty = np.ones((param.local_size_height, param.local_size_width))
        else:
            m, m_uncertainty = res
            self.local_m = self.get_local_map(m,self.current_pose )
            self.local_m_uncertainty = self.get_local_map(m_uncertainty, self.current_pose )
        
        return self.local_m, self.local_m_uncertainty, self.current_pose

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

    def ToSAC_step(self, sub_goal):
        '''
        与RL step的接口
        '''
        x_new = sub_goal[0]
        y_new = sub_goal[1]
        yaw_new = sub_goal[2]

        m, m_uncertainty = self.lidar.update([x_new, y_new, yaw_new])
        self.current_pose = np.array([x_new, y_new, yaw_new])
        
        self.local_m = self.get_local_map(m,self.current_pose )
        self.local_m_uncertainty = self.get_local_map(m_uncertainty, self.current_pose )

        # 是否碰撞
        collision = self.is_collision(self.current_pose)
        
        return self.local_m, self.local_m_uncertainty, self.current_pose, collision



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
        self.L0 = np.zeros((self.M, self.N))                # Prior log-odds
        self.L = np.zeros((self.M, self.N))
        self.m = np.full((self.M, self.N), 0.5)             # 初始化感知概率地图为0.5
        self.ms = []                                        # 存储每一帧的感知概率地图
        self.m_uncertainty = np.full((self.M, self.N),1)    # 初始化不确定性地图
        self.m_uncertaintys = []                            # 存储每一帧的不确定性地图

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


    def inverse_scanner(self, X, meas_r):
        """
        返回 m(i,j) : 当前帧观测下，每个格子的“占用概率” (0.3/0.5/0.7)
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
        m_local = np.full_like(r, 0.5, dtype=float)

        # 对每条激光束判断
        for k, angle in enumerate(self.meas_phi):

            mask_angle = np.abs(phi - angle) <= self.beta / 2

            mask_occ = mask_angle & (np.abs(r - meas_r[k]) <= self.alpha / 2)
            mask_free = mask_angle & (r < meas_r[k])

            m_local[mask_occ] = 0.7
            m_local[mask_free] = 0.3

        # 写回全局
        h, w = m_local.shape
        self.m[y_min:y_min+h, x_min:x_min+w] = m_local
                    
        return self.m
    
    def get_uncertainty_map(self, X):
        """
        生成不确定性地图
        """
        self.m_uncertainty = -(self.m * np.log(self.m) + (1 - self.m) * np.log(1 - self.m))
        self.m_uncertainty /= np.log(2)
        self.m_uncertainty = np.clip(self.m_uncertainty, 1e-6, 1 - 1e-6)

        self.m_uncertaintys.append(self.m_uncertainty)          

        return self.m_uncertainty

    def generate_probability_map(self, X):
        """
        生成感知概率地图
        """
        meas_rs = []
        invmods = []
        
        meas_r = self.get_ranges(X)
        meas_rs.append(meas_r)
        invmod = self.inverse_scanner(X,meas_r)
        invmod = np.clip(invmod, 0.1, 0.99)  # 避免log(0)
        invmods.append(invmod)
        
        # 计算逆模型对应的 log-odds
        inv_logodds = np.log(invmod / (1 - invmod))
        # 更新
        self.L = self.L + inv_logodds - self.L0
        # Clamp 避免爆炸
        self.L = np.clip(self.L, -20, 20)

        # Calculate a grid of probabilities from the log odds.
        self.m = np.divide(np.exp(self.L), np.add(1, np.exp(self.L)))
        self.m = np.clip(self.m, 1e-6, 1 - 1e-6)

        self.ms.append(self.m)

        return self.m

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
        m_uncertainty = self.get_uncertainty_map(X_clipped)

        return m, m_uncertainty
    
def main():
    '''
    测试代码
    '''
    # 读取地图
    map_converter = ReadGridMap()
    big_map,small_map = map_converter.convert("/home/fmt/decision_making/sb3_SAC/map/map_basic.png")
    lidar = Lidar(big_map)

    # action输入: 车辆/雷达状态 [x, y, theta]
    for x in range(1,10):
        for y in range(1):
            for yaw in [math.pi/3]:
                m,m_uncertainty = lidar.update([x,y,yaw])

    # 计算不确定性地图
    x,y,yaw = 50,50,math.pi/3
    uncertainty_map = lidar.get_uncertainty_map([x,y,yaw])

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    img0 = axes[0].imshow(m, cmap='gray_r', origin='lower', vmin=0, vmax=1)
    fig.colorbar(img0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("Occupancy map")

    img1 = axes[1].imshow(big_map, cmap='gray_r', origin='lower', vmin=0, vmax=1)
    fig.colorbar(img1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Original grid map")

    # 风格1 
    img2 = axes[2].imshow(uncertainty_map, cmap='jet', origin='lower', vmin=0, vmax=1)
    fig.colorbar(img2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Uncertainty map")

    # 风格2
    # sns.heatmap(
    #     uncertainty_map,
    #     ax=axes[2],
    #     cmap='YlOrRd',
    #     square=True,
    #     cbar_kws={"shrink": 0.8},
    #     linewidths=0.0
    # )
    print("uncertainty_map:", uncertainty_map.shape)
    print("map:", m.shape)
    print("small_map:", small_map.shape)
    print("big_map:", big_map.shape)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    '''
    目前仍存在的问题
    1.环境是否太简单，如何让其更真实一些
    2.感知不确定性如何衡量，现在用信息熵衡量能否满足要求
    3.如何体现越野场景（地形起伏大，只用一个地图来代替是否可行）
    4.传感器模型没有噪声，如何体现不确定性（针对当前技术，传感器噪声是否可以忽略，调研当前传感器
        技术水平，评估lidar模型是否合理）
    5.如何融合点云数据
    6.保存环境数据进行可视化
    '''
    pass
