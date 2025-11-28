import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from read_grid_map import ReadGridMap
import HybridAstarPlanner.Hybrid_Astar as planning 
import time

class param:
    # Hybrid A* 参数
    XY_RESO = 1.0                   # 网格地图分辨率
    YAW_RESO = np.deg2rad(15.0)     # 航向角分辨率
    WB = 3.5                        # 车辆的轴距 (WheelBase)
    MOVE_STEP = 0.4                 # 插值分辨率，表示在路径中每个节点之间的距离
    # 局部地图尺寸
    local_size_width = 64
    local_size_height = 64
    # reward 参数
    REACH_GOAL_REWARD = 100.0       # 到达目标奖励
    COLLISION_PENALTY = -100.0      # 碰撞惩罚
    STEP_PENALTY = -1.0             # 每步惩罚
    DISTANCE_WEIGHT = 1.0           # 距离权重
    YAW_WEIGHT = -0.5               # 航向角权重
    EXPLORE_GAIN = 1.0              # 探索奖励增益

class interface2RL:
    def __init__(self, global_map, init_pose = [0.0, 0.0, 0.0]):
        # 创建调用对象
        self.global_map = global_map
        self.lidar = Lidar(global_map)

        self.local_m = np.zeros((param.local_size_height, param.local_size_width))
        self.local_m_uncertainty = np.zeros((param.local_size_height, param.local_size_width))

        self.current_pose = np.array(init_pose, dtype=np.float32)  # 车辆当前坐标和航向

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
        # print(x)
        # print(y)
        # print(yaw)
    
        # 在每个时刻绘制车辆的轨迹和障碍物
        # for k in range(len(x)):
        #     plt.cla()         # 清空当前图形 
        #     plt.plot(ox, oy, "sk")   # 绘制障碍物和规划路径
        #     plt.plot(x, y, linewidth=1.5, color='r')
    
        #     # 计算当前时刻的方向和转向角
        #     if k < len(x) - 2:
        #         dy = (yaw[k + 1] - yaw[k]) / param.MOVE_STEP
        #         steer = planning.rs.pi_2_pi(math.atan(-param.WB * dy / direction[k]))
        #     else:
        #         steer = 0.0
    
            # 绘制车辆
            # planning.draw_car(gx, gy, gyaw0, 0.0, 'dimgray')   # 绘制目标点
            # planning.draw_car(x[k], y[k], yaw[k], steer)       # 绘制当前车辆状态
            # plt.title("Hybrid A*")
            # plt.axis("equal")
            # plt.pause(0.0001)
    
        # plt.show()
        # print("Done!")
        return x,y, yaw, direction

    def ToSAC_reset(self):
        '''
        与RL reset的接口
        '''
        # m, m_uncertainty = self.lidar.update([self.current_pose[0], 
        #                                       self.current_pose[1], 
        #                                       self.current_pose[2]])
        res = self.lidar.update([self.current_pose[0], 
                         self.current_pose[1], 
                         self.current_pose[2]])
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

        # x,y,yaw,_ = self.GetPath(self.current_pose[0], self.current_pose[1], self.current_pose[2],
        #                          sub_goal[0], sub_goal[1], sub_goal[2])
        path_result = self.GetPath(self.current_pose[0], self.current_pose[1], self.current_pose[2],
                                  sub_goal[0], sub_goal[1], sub_goal[2])
        if path_result is None:
            x_new, y_new, yaw_new = self.current_pose
            collision = True  # 也可以改成 False，看你想怎么奖励
        else:
            x, y, yaw, _ = path_result
            if len(x) < 2:
                # 规划路径太短，相当于没动
                x_new, y_new, yaw_new = self.current_pose
            else:
                # 只走一步
                x_new = x[1]
                y_new = y[1]
                yaw_new = yaw[1]
        

        # # 只走一步，更新地图
        # if x is None or len(x) < 2:
        #     # 可以选：停在原地 / 给个惩罚 / done = True
        #     x_new, y_new, yaw_new = self.current_pose
        # else:
        #     x_new = x[1]
        #     y_new = y[1]
        #     yaw_new = yaw[1]

        m, m_uncertainty = self.lidar.update([x_new, y_new, yaw_new])
        self.current_pose = np.array([x_new, y_new, yaw_new])
        
        self.local_m = self.get_local_map(m,self.current_pose )
        self.local_m_uncertainty = self.get_local_map(m_uncertainty, self.current_pose )

        # collision
        collision = self.is_collision(self.current_pose)
        
        return self.local_m, self.local_m_uncertainty, self.current_pose, collision



class Lidar:
    def __init__(self, true_map, reso=1):
        self.true_map = true_map
        self.reso = reso
        # Parameters for the sensor model.
        self.meas_phi = np.arange(-0.4, 0.4, 0.05)
        self.rmax = 30 # Max beam range.
        self.alpha = 1 # Width of an obstacle (distance about measurement to fill in).
        self.beta = 0.05 # Angular width of a beam.
        (self.M, self.N) = np.shape(true_map)
        self.L0 = np.zeros((self.M, self.N))  # Prior log-odds
        self.L = np.zeros((self.M, self.N))
        self.m = np.full((self.M, self.N), 0.5)  # 初始化感知概率地图为0.5
        self.ms = []  # 存储每一帧的感知概率地图
        self.m_uncertainty = np.full((self.M, self.N),1)  # 初始化不确定性地图
        self.m_uncertaintys = []  # 存储每一帧的不确定性地图

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
        # x_max = x_ind + self.rmax / self.reso
        # y_max = y_ind + self.rmax / self.reso

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
        
        # x_ind, y_ind, theta = X[0]/self.reso, X[1]/self.reso, X[2]
        # x_max = min(self.N, x_ind + self.rmax / self.reso)
        # y_max = min(self.M, y_ind + self.rmax / self.reso)
        # x_min = max(0, x_ind - self.rmax / self.reso)
        # y_min = max(0, y_ind - self.rmax / self.reso)

        # self.m = np.clip(self.m, 1e-6, 1 - 1e-6)  # 避免log(0)


        # for i in range(int(y_min), int(y_max)):
        #     for j in range((int(x_min)), int(x_max)):
        #         self.m_uncertainty[i,j] = (-(self.m[i,j] * np.log(self.m[i,j]) + (1 - self.m[i,j]) * np.log(1 - self.m[i,j])))/np.log(2) # 计算不确定性(信息熵)
        
        m = np.clip(self.m, 1e-6, 1 - 1e-6)
        self.m_uncertainty = -(m * np.log(m) + (1 - m) * np.log(1 - m))
        self.m_uncertainty /= np.log(2)

        self.m_uncertaintys.append(self.m_uncertainty) # 存储当前帧的不确定性地图

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
        invmod = np.clip(invmod, 1e-6, 1 - 1e-6)  # 避免log(0)
        invmods.append(invmod)
        # Calculate and update the log odds of our occupancy grid, given our measured occupancy probabilities from the inverse model.
        self.L = np.log(np.divide(invmod, np.subtract(1, invmod))) + self.L - self.L0 # concernnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
        
        # Calculate a grid of probabilities from the log odds.
        self.m = np.divide(np.exp(self.L), np.add(1, np.exp(self.L)))

        self.ms.append(self.m)

        return self.m

        # Main simulation loop.
        # for t in range(1, len(time_steps)):
        #     # Perform robot motion.
        #     move = np.add(x[0:2, t-1], u[:, u_i]) 
        #     # If we hit the map boundaries, or a collision would occur, remain still.
        #     if (move[0] >= M - 1) or (move[1] >= N - 1) or (move[0] &lt;= 0) or (move[1] &lt;= 0) or true_map[int(round(move[0])), int(round(move[1]))] == 1:
        #         x[:, t] = x[:, t-1]
        #         u_i = (u_i + 1) % 4
        #     else:
        #         x[0:2, t] = move
        #     x[2, t] = (x[2, t-1] + w[t]) % (2 * math.pi)
            
        #     # Gather the measurement range data, which we will convert to occupancy probabilities
        #     meas_r = get_ranges(true_map, x[:, t], meas_phi, rmax)
        #     meas_rs.append(meas_r)
            
        #     # Given our range measurements and our robot location, apply inverse scanner model
        #     invmod = inverse_scanner(M, N, x[0, t], x[1, t], x[2, t], meas_phi, meas_r, rmax, alpha, beta)
        #     invmods.append(invmod)
            
        #     # Calculate and update the log odds of our occupancy grid, given our measured occupancy probabilities from the inverse model.
        #     L = np.log(np.divide(invmod, np.subtract(1, invmod))) + L - L0
            
            
        #     # Calculate a grid of probabilities from the log odds.
        #     m = np.divide(np.exp(L), np.add(1, np.exp(L)))
        #     ms.append(m)

    def update(self, X):
        """
        更新感知概率地图和不确定性地图
        """
        if X[0]<0 or X[0]>=self.N*self.reso or X[1]<0 or X[1]>=self.M*self.reso:
            print("Warning: Lidar position out of bounds.")
            return None
        m = self.generate_probability_map(X)
        m_uncertainty = self.get_uncertainty_map(X)

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

    # 风格2 
    img2 = axes[2].imshow(uncertainty_map, cmap='jet', origin='lower', vmin=0, vmax=1)
    fig.colorbar(img2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Uncertainty map")

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
    inte = interface2RL()
    inte.GetPath()
