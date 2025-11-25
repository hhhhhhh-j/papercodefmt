import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging
import seaborn as sns
# from lidar import Lidar

class Map():
    """带感知不确定性的栅格地图类"""
    
    def __init__(self, width, height,start=None,goal=None):
        """
        初始化地图
        :param width: 地图宽度
        :param height: 地图高度
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))  # 真实地图：0表示空地，1表示障碍物
        self.perception_map = np.full((height, width), 0.5)  # 感知地图：0.5表示未知
        self.perception_certainty = np.zeros((height, width))  # 感知确定性：0-1
        self.uncertainty_map = np.ones((height, width))  # 不确定性地图：初始为最大不确定性
        self.start = start
        self.goal = goal
        
    def create_map(self, obstacle_positions=None):
        """
        创建地图及障碍物
        :param obstacle_positions: 障碍物位置列表 [(x1, y1), (x2, y2), ...]
        """
        # 清空地图（原始地图、感知map、确定性map）
        self.grid = np.zeros((self.height, self.width))
        self.perception_map = np.full((self.height, self.width), 0.5)
        self.perception_certainty = np.zeros((self.height, self.width))
        
        # 添加障碍物
        if obstacle_positions:
            for pos in obstacle_positions:
                x, y = pos
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[y][x] = 1
        return self.grid
                    
    def add_obstacle(self, x, y):
        """添加单个障碍物"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = 1
            
    def add_rectangle_obstacle(self, x, y, width, height):
        """添加矩形障碍物"""
        for i in range(y, min(y + height, self.height)):
            for j in range(x, min(x + width, self.width)):
                self.grid[i][j] = 1
                
    def add_circle_obstacle(self, center_x, center_y, radius):
        """添加圆形障碍物"""
        for i in range(self.height):
            for j in range(self.width):
                if (j - center_x)**2 + (i - center_y)**2 <= radius**2:
                    self.grid[i][j] = 1
                    
    def remove_obstacle(self, x, y):
        """移除障碍物"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = 0
            
    def is_obstacle(self, x, y):
        """检查某位置是否为障碍物（基于真实地图）"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x] == 1
        return True
    
    # def perceive(self, agent_x, agent_y, sensor_range=5, 
    #              false_positive_rate=0.1, false_negative_rate=0.15):
    #     """
    #     在agent位置进行感知，更新感知地图
    #     :param agent_x: agent的x坐标
    #     :param agent_y: agent的y坐标
    #     :param sensor_range: 传感器范围
    #     :param false_positive_rate: 误检率（将空地识别为障碍物）
    #     :param false_negative_rate: 漏检率（将障碍物识别为空地）
    #     """
    #     for i in range(self.height):
    #         for j in range(self.width):
    #             # 计算距离
    #             distance = np.sqrt((j - agent_x)**2 + (i - agent_y)**2)
                
    #             if distance <= sensor_range:
    #                 # 真实状态
    #                 is_obstacle = self.grid[i][j] == 1
                    
    #                 # 根据距离调整感知准确度（距离越远越不准确）-------------------------------------------后续需要更精准
    #                 distance_factor = 1 - (distance / sensor_range) * 0.5 # -----------------------------------
    #                 adjusted_fp = false_positive_rate / distance_factor # 误检率/-------------------------------
    #                 adjusted_fn = false_negative_rate / distance_factor # 漏检率/-------------------------------
                    
    #                 # 模拟传感器噪声
    #                 rand = np.random.random()
                    
    #                 if is_obstacle:
    #                     # 真实是障碍物
    #                     if rand < adjusted_fn:
    #                         perceived = 0  # 漏检
    #                     else:
    #                         perceived = 1  # 正确检测
    #                 else:
    #                     # 真实是空地
    #                     if rand < adjusted_fp:
    #                         perceived = 1  # 误检
    #                     else:
    #                         perceived = 0  # 正确检测
                    
    #                 # 使用贝叶斯更新感知置信度
    #                 self._update_perception(j, i, perceived, distance_factor)
    
    # def _update_perception(self, x, y, observation, confidence): # ----------------------------------------------------------
    #     """
    #     使用贝叶斯方法更新感知地图
    #     :param x: x坐标
    #     :param y: y坐标
    #     :param observation: 观察结果（0或1）
    #     :param confidence: 观察置信度（0-1）
    #     """
    #     # 当前先验概率
    #     prior = self.perception_map[y][x]
        
    #     # 似然函数（观察到该结果的概率）
    #     if observation == 1:
    #         likelihood = confidence
    #     else:
    #         likelihood = 1 - confidence
        
    #     # 贝叶斯更新
    #     if observation == 1:
    #         # 观察到障碍物
    #         posterior = (prior * likelihood) / (prior * likelihood + (1 - prior) * (1 - likelihood))
    #     else:
    #         # 观察到空地
    #         posterior = (prior * likelihood) / (prior * likelihood + (1 - prior) * (1 - likelihood))
        
    #     # 更新感知地图
    #     self.perception_map[y][x] = posterior

        
    
    # def is_perceived_obstacle(self, x, y, threshold=0.6):
    #     """
    #     基于感知地图判断是否为障碍物
    #     :param x: x坐标
    #     :param y: y坐标
    #     :param threshold: 判定阈值
    #     :return: True表示感知为障碍物
    #     """
    #     if 0 <= x < self.width and 0 <= y < self.height:
    #         return self.perception_map[y][x] > threshold
    #     return True
    
    # def get_perception_uncertainty(self, x, y):
    #     """
    #     获取某位置的感知不确定性
    #     :param x: x坐标
    #     :param y: y坐标
    #     :return: 不确定性值（0-1，越大越不确定）
    #     """
    #     if 0 <= x < self.width and 0 <= y < self.height:
    #         prob = self.perception_map[y][x]
    #         # 熵作为不确定性度量
    #         if prob == 0 or prob == 1:
    #             return 0
    #         return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)
    #     return 1.0
    
    def visualize(self, agent_pos=None, start=None, goal=None, path=None, 
                  show_perception=True, show_uncertainty=True):
        """
        可视化地图
        :param agent_pos: agent位置 (x, y)
        :param start: 起点坐标 (x, y)
        :param goal: 终点坐标 (x, y)
        :param path: 路径点列表
        :param show_perception: 是否显示感知地图
        :param show_uncertainty: 是否显示不确定性地图
        """
        if show_perception:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes = [axes[0], axes[1]]
        
        # 1. 真实地图
        axes[0].imshow(self.grid, cmap='binary', origin='upper')
        axes[0].set_title('Ground Truth Map', fontsize=14, fontweight='bold')
        self._add_grid_and_markers(axes[0], agent_pos, start, goal, path)
        
        # 2. 感知地图
        if show_perception:
            # 创建自定义颜色映射：(空地)-(未知)-(障碍物)
            colors = ['white', 'gray', 'black']
            n_bins = 100
            cmap = LinearSegmentedColormap.from_list('perception', colors, N=n_bins)
            
            im = axes[1].imshow(self.perception_map, cmap=cmap, origin='upper', vmin=0, vmax=1)
            axes[1].set_title('Perception Map', 
                            fontsize=14, fontweight='bold')
            cbar = plt.colorbar(im, ax=axes[1])
            # 设置颜色条刻度
            cbar.set_ticks([0.0, 0.5, 1.0])
            cbar.set_ticklabels(['Free', 'Unknown', 'Obstacle'])
            self._add_grid_and_markers(axes[1], agent_pos, start, goal, path)
        
        # 3. 不确定性地图
        if show_uncertainty and axes[2] is not None:
            for i in range(self.height):
                for j in range(self.width):
                    self.uncertainty_map[i][j] = self.get_perception_uncertainty(j, i)
            
            # 风格1
            # im = axes[2].imshow(
            #     self.uncertainty_map,
            #     cmap='cividis',
            #     origin='upper'
            # )

            # 风格2
            sns.heatmap(
                self.uncertainty_map,
                ax=axes[2],
                cmap='YlOrRd',
                square=True,
                cbar_kws={"shrink": 0.8},
                linewidths=0.0
            )

            # 风格3
            # im = axes[2].imshow(
            # self.uncertainty_map,
            # cmap='coolwarm',
            # origin='upper'

            axes[2].grid(color='gray', linestyle='--', linewidth=0.5)
            axes[2].set_title('Perception Uncertainty', 
                            fontsize=14, fontweight='bold')
            # plt.colorbar(im, ax=axes[2], label='Uncertainty')
            self._add_grid_and_markers(axes[2], agent_pos, start, goal, path)
        
        plt.tight_layout()
        plt.show()
    
    def _add_grid_and_markers(self, ax, agent_pos, start, goal, path):
        """添加网格线和标记"""
        # 绘制网格线
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.3, alpha=0.5)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.3, alpha=0.5)
        
        # 标记agent
        if agent_pos:
            circle = plt.Circle((agent_pos[0], agent_pos[1]), 0.4, 
                              color='yellow', fill=True, zorder=5)
            ax.add_patch(circle)
            ax.plot(agent_pos[0], agent_pos[1], 'ko', markersize=8, zorder=6)
            
        # 标记起点
        if start:
            ax.plot(start[0], start[1], 'go', markersize=12, label='Start', zorder=4)
            
        # 标记终点
        if goal:
            ax.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal', zorder=4)
            
        # 绘制路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.6, label='Path', zorder=3)
            
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(self.height - 0.5, -0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(False)


# 使用示例
if __name__ == "__main__":
    # 定义起点、终点和agent位置
    start = (1, 1)
    goal = (18, 18)
    agent_pos = (5, 10)
    game_map = Map(30, 30,start,goal)
    
    # 添加障碍物
    game_map.add_rectangle_obstacle(8, 5, 4, 8)
    game_map.add_circle_obstacle(15, 5, 2)
    game_map.add_rectangle_obstacle(3, 15, 5, 3)
    
    # Agent进行多次感知（模拟移动和重复观测）
    print("执行感知...")
    for x in range(1, 15):
        for y in range(1, 15):
            game_map.perceive(x, y, sensor_range=6)
    # game_map.perceive(agent_pos[0], agent_pos[1], sensor_range=6)
    # game_map.perceive(8, 8, sensor_range=6)
    # game_map.perceive(10, 12, sensor_range=5)
    # game_map.perceive(12, 15, sensor_range=6)
    
    # 可视化：显示真实地图、感知地图和不确定性
    game_map.visualize(agent_pos=agent_pos, start=start, goal=goal, 
                      show_perception=True, show_uncertainty=True)
    
    # 测试感知判断
    # print(f"\n位置({agent_pos[0]}, {agent_pos[1]}):")
    # print(f"  真实状态: {'障碍物' if game_map.is_obstacle(agent_pos[0], agent_pos[1]) else '空地'}")
    # print(f"  感知概率: {game_map.perception_map[agent_pos[1]][agent_pos[0]]:.2f}")
    # print(f"  感知判断: {'障碍物' if game_map.is_perceived_obstacle(agent_pos[0], agent_pos[1]) else '空地'}")
    # print(f"  point({agent_pos[0]:.2f},{agent_pos[1]:.2f})的不确定性: {game_map.get_perception_uncertainty(agent_pos[0], agent_pos[1]):.2f}")
