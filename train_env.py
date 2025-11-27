import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from read_grid_map import ReadGridMap

class interface2RL:
    def __init__(self):
        pass

    def GetPath(sub_goal_x, sub_goal_y, sub_goal_yaw):
        '''
        获取行动路径
        '''
        
    pass

    def get_local_occupy_map():
        '''
        获取局部地图
        '''
        pass

    def get_local_uncertainty_map():
        '''
        获取局部不确定性地图
        '''
        pass

    def ToSacEnv_reset():
        '''
        与RL reset的接口
        '''
        pass

    def ToSacEnv_step():
        '''
        与RL step的接口
        '''
        pass



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
        self.ms = []  # 存储每一帧的感知概率地图
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
                if (xi <= 0 or xi >= self.M-1 or yi <= 0 or yi >= self.N-1):
                    meas_r[i] = r
                    break
                # If in the map, but hitting an obstacle, set the measurement range
                # and stop ray tracing.
                elif self.true_map[int(round(xi)), int(round(yi))] == 1:
                    meas_r[i] = r
                    break
                    
        return meas_r


    def inverse_scanner(self, X, meas_r):
        """
        返回 m(i,j) : 当前帧观测下，每个格子的“占用概率” (0.3/0.5/0.7)
        alpha:
        beta:
        """
        x_ind, y_ind, theta = X[0]/self.reso, X[1]/self.reso, X[2]
        x_max = x_ind + self.rmax / self.reso
        y_max = y_ind + self.rmax / self.reso
        m = np.full((self.M, self.N), 0.5)
        for i in range(int(x_ind), min(self.M, int(x_max))):
            for j in range((int(y_ind)), min(self.N, int(y_max))):
                # Find range and bearing relative to the input state (x, y, theta).
                r = math.sqrt((i - x_ind)**2 + (j - y_ind)**2)
                phi = (math.atan2(j - y_ind, i - x_ind) - theta + math.pi) % (2 * math.pi) - math.pi
                
                # Find the range measurement associated with the relative bearing.
                k = np.argmin(np.abs(np.subtract(phi, self.meas_phi)))
                
                # If the range is greater than the maximum sensor range, or behind our range
                # measurement, or is outside of the field of view of the sensor, then no
                # new information is available.
                if (r > min(self.rmax, meas_r[k] + self.alpha / 2.0)) or (abs(phi - self.meas_phi[k]) > self.beta / 2.0):
                    m[i, j] = 0.5
                
                # If the range measurement lied within this cell, it is likely to be an object.
                elif (meas_r[k] < self.rmax) and (abs(r - meas_r[k]) < self.alpha / 2.0):
                    m[i, j] = 0.7
                
                # If the cell is in front of the range measurement, it is likely to be empty.
                elif r < meas_r[k]:
                    m[i, j] = 0.3
                    
        return m
    
    def get_uncertainty_map(self, X):
        """
        生成不确定性地图
        """
        
        x_ind, y_ind, theta = X[0]/self.reso, X[1]/self.reso, X[2]
        x_max = min(self.M, x_ind + self.rmax / self.reso)
        y_max = min(self.N, y_ind + self.rmax / self.reso)
        x_min = max(0, x_ind - self.rmax / self.reso)
        y_min = max(0, y_ind - self.rmax / self.reso)
        m = self.generate_probability_map(X)
        m = np.clip(m, 1e-6, 1 - 1e-6)  # 避免log(0)
        m_uncertainty = np.zeros_like(m)

        for i in range(int(x_min), int(x_max)):
            for j in range((int(y_min)), int(y_max)):
                m_uncertainty[i,j] = m[i,j] * np.log(m[i,j]) + (1 - m[i,j]) * np.log(1 - m[i,j]) # 计算不确定性(信息熵)
        
        self.m_uncertaintys.append(m_uncertainty) # 存储当前帧的不确定性地图

        return -m_uncertainty

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
        m = np.divide(np.exp(self.L), np.add(1, np.exp(self.L)))

        self.ms.append(m)

        return m

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

def main():
    '''
    测试代码
    '''

    # 生成地图
    # map = Map(50,50)
    # test_map = map.create_map()
    # map.add_rectangle_obstacle(8, 5, 4, 8)
    # map.add_circle_obstacle(15, 5, 2)
    # map.add_rectangle_obstacle(3, 15, 5, 3)

    # 读取地图
    map_converter = ReadGridMap()
    big_map,small_map = map_converter.convert("/home/fmt/decision_making/sb3_SAC/map/map_basic.png")
    lidar = Lidar(big_map)

    # action输入: 车辆/雷达状态 [x, y, theta]
    for x in range(1,20):
        for y in range(1,20):
            for yaw in [math.pi/3]:
                m = lidar.generate_probability_map([x,y,yaw])

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
    main()
