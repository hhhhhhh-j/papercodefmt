# 构建topo map加入到状态空间
# 需要一个节点类 Node
# 以及一个topo map类 TopoMap
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import math
import sys
import itertools
from utils.draw import draw_agent
from heapq import heappush, heappop

UNKNOWN = -1
FREE    = 0
OCC     = 1



class param:
    # Hybrid A* 参数
    XY_RESO = 1.0                               # 一个格子代表多少m
    YAW_RESO = np.deg2rad(15.0)                 # 航向角分辨率
    WB = 3.5                                    # 车辆的轴距 (WheelBase)
    max_velocity = 10.0                         # 最大速度 m/s

    # 地图尺寸
    local_size_width = 64
    local_size_height = 64
    global_size_width = 256
    global_size_height = 256
    max_dist = np.sqrt(global_size_width**2 + global_size_height**2)
    max_dist_local = np.sqrt(local_size_width**2 + local_size_height**2)
    frontier_k = 8                              # 选择前 k 个 frontier clusters
    SEED = None                                 # 随机地图种子

    # action 缩放系数
    RATIO_throttle = 4                          # x缩放系数
    RATIO_steer = math.pi / 4                     # yaw缩放系数
    dt = 0.3                                    # 时间步长
    L = 2.5                                     # 车辆轴距

    # reward 系数
    REACH_GOAL_WEIGHT = 100.0                    # 到达目标奖励
    COLLISION_WEIGHT = -50.0                    # 碰撞惩罚
    STEP_PENALTY_WEIGHT = 1.0                  # step惩罚
    DISTANCE_WEIGHT = 5.30                      # 距离权重
    YAW_WEIGHT = 0.05                            # 航向角权重
    EXPLORE_GAIN_WEIGHT = 0.5                   # 探索奖励增益
    # REVERSE_WEIGHT = 1.0                        # 倒车惩罚
    # FOWARD_WEIGHT = 0.5                         # 前进奖励
    VISIT_PENALTY_WEIGHT = 0.5                  # 访问频次惩罚系数
    RISK_PENALTY_WEIGHT = 0.05                  # 风险惩罚系数
    NOPATH_PENALTY_WEIGHT = 1.0                 # 无路径惩罚系数

    # 111
    PATCH_SIZE_beliefmap = 256                  # 全局地图patch大小

    # lidar参数
    RESO = 1
    MEAS_PHI = np.arange(-0.6, 0.6, 0.05)
    RMAX = 30                                   # Max beam range.
    ALPHA = 1                                   # Width of an obstacle (distance about measurement to fill in).
    BETA = 0.05                                 # Angular width of a beam.
    gamma = 1.0000000000000                     # 遗忘指数

class Frontier:
    def __init__(self, local_occ, local_unk, local_free, robot_ij, action, k=8):
        self.k = k                                                                          # 选择前 k 个 frontier clusters
        self.size_norm_factor = 10.0 / (param.local_size_width * param.local_size_height)   # cluster size 归一化因子
        self.dist_norm_factor = 1.0 / param.max_dist_local
        self.risk_norm_factor = 1.0                                                         # 假定 risk 在 [0,1] 之间
        H, W = local_free.shape
        self.grid = np.full((H, W), UNKNOWN)
        self.grid[local_occ >= 0.7] = OCC
        self.grid[local_free >= 0.7] = FREE
        self.robot_ij = robot_ij                                                            # 机器人在 grid 中的位置 (i,j)
        self.action = action                                                                # 用于评分的权重参数

    def compute_frontier_mask(self):
        free = (self.grid == FREE).astype(np.uint8)
        unk  = (self.grid == UNKNOWN).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        unk_near = cv2.dilate(unk, kernel, iterations=1)  # unknown 周围一圈

        frontier = (free & unk_near).astype(np.uint8)
        return frontier  # 0/1


    def cluster_frontiers(self, frontiers, min_size=3):
        # labels: 0=背景, 1..num_labels-1 = 每个连通域
        num_labels, labels = cv2.connectedComponents(frontiers, connectivity=8)

        clusters = []
        for lab in range(1, num_labels):
            ys, xs = np.where(labels == lab)   # ys=i, xs=j
            if ys.size < min_size:
                continue
            cells = np.stack([ys, xs], axis=1)  # (N,2) 每行 (i,j)
            clusters.append(cells)

        return clusters


    def summarize_clusters_rep_points(self, clusters):
        ri, rj = self.robot_ij
        infos = []

        for cells in clusters:
            # centroid (浮点)
            ic = float(cells[:, 0].mean())
            jc = float(cells[:, 1].mean())

            # rep：离机器人最近的格子
            di = cells[:, 0] - ri
            dj = cells[:, 1] - rj
            k = int(np.argmin(di*di + dj*dj))
            rep_ij = (int(cells[k, 0]), int(cells[k, 1]))

            infos.append({
                "cells": cells,              
                "size": int(cells.shape[0]),
                "centroid_ij": (ic, jc),
                "rep_ij": rep_ij
            })

        return infos

    def select_topk(self, infos, risk_map=None):
        ri, rj = self.robot_ij

        def score(info):                                        # use action to score
            i, j = info["rep_ij"]
            dist = np.hypot(i - ri, j - rj) + 1e-6
            size = info["size"]

            risk = 0.0
            if risk_map is not None:
                risk = float(risk_map[i, j])

            # 分数越大越好：偏好“大cluster、近一些、低风险”
            # score_ = self.action[0] * size - self.action[1] * dist - self.action[2] * risk
            # 归一化
            size_norm = size * self.size_norm_factor
            dist_norm = dist * self.dist_norm_factor
            risk_norm = risk * self.risk_norm_factor
            score_ = self.action[0] * size_norm - self.action[1] * dist_norm - self.action[2] * risk_norm
            return score_

        infos_sorted = sorted(infos, key=score, reverse=True)
        topk = infos_sorted[:self.k]

        # padding（不足 K 用 None 填满）
        while len(topk) < self.k:
            topk.append(None)

        return topk
        
class Render:
    def __init__(self):
        # render可视化参数
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 3, figsize=(14,5))

    def _draw_local_map(self, local_map, agent_x, agent_y, agent_yaw):
        self.ax[0].clear()

        x_min = agent_x - (param.local_size_width)/2
        x_max = agent_x + (param.local_size_width)/2
        y_min = agent_y - (param.local_size_height)/2
        y_max = agent_y + (param.local_size_height)/2

         # --- 图：local map ---
        self.ax[0].clear()
        self.ax[0].imshow(local_map, cmap="gray_r", origin="lower",
                          extent=[ x_min, x_max, y_min, y_max])
        self.ax[0].set_title("Local Map")

        # draw agent
        draw_agent(self.ax[0], agent_x, agent_y, agent_yaw)

    def _draw_local_map_uncertainty(self, local_map_uncertainty, agent_x, agent_y, agent_yaw):
        self.ax[1].clear()

        x_min = agent_x - (param.local_size_width)/2
        x_max = agent_x + (param.local_size_width)/2
        y_min = agent_y - (param.local_size_height)/2
        y_max = agent_y + (param.local_size_height)/2

        # --- 图：uncertainty map ---
        self.ax[1].clear()
        self.ax[1].imshow(local_map_uncertainty, cmap="turbo", vmin=0, vmax=1,origin="lower",
                          extent=[x_min, x_max, y_min, y_max], # interpolation="bilinear", 
                          interpolation="bicubic")
        self.ax[1].set_title("Uncertainty Map")

        # draw agent
        draw_agent(self.ax[1], agent_x, agent_y, agent_yaw)

    def _draw_belief_map(self, belief_map):
        # belief_map 为空就不画
        if len(belief_map) == 0:
            return

        H = param.global_size_height      # 256
        W = param.global_size_width       # 256

        # 全局栅格，默认未知 0.5
        grid = np.ones((H, W)) * 0.5

        # belief_map 的 key 约定为 (x, y) = (列索引, 行索引)
        for (x, y), b in belief_map.items():
            # 防止越界
            if 0 <= x < W and 0 <= y < H:
                grid[y, x] = b["occ"]

        # 空间平滑（可选）
        grid_smooth = nd.gaussian_filter(grid, sigma=1.0)

        self.ax[2].clear()
        img = self.ax[2].imshow(
            grid_smooth,
            cmap="gray_r",
            origin="lower",               # y 轴向上
            vmin=0,
            vmax=1,
            extent=[0, W, 0, H]           # 和 global_map / agent 坐标保持一致
        )
        self.ax[2].set_title("Belief Map (Dense)")

    def flush(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Planner:
    # 注意: y, x 解包
    F, H, NUM, G, POS, OPEN, VALID, PARENT = range(8)

    def __init__(self):
        self.local_map = None
        self.goal_pos = None
        self.nodes = {}
    
    def __contains__(self, pos):
        y, x = pos
        return 0 <= y < param.local_size_height \
            and 0 <= x < param.local_size_width \
            and self.local_map[y,x] <= 0.5

    def main_workflow(self, local_map, goal):
        self.local_map = local_map
        self.goal_pos = goal
        path = self.astar(
            start_pos=(param.local_size_height // 2, param.local_size_width // 2),
            neighbors=self.neighbors,
            goal = self.reach_goal,
            start_g=0,
            cost=self.cost,
            heuristic=self.heuristic,
            limit=10000,
            debug=self.debug)
        return path

    def neighbors(self, pos):
        y, x = pos
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),
                       (1, 0), (1, 1)):
            if (y + dy, x + dx) in self:
                yield y + dy, x + dx

    def reach_goal(self, pos):
        return pos == self.goal_pos

    def cost(self, from_pos, to_pos):
        from_y, from_x = from_pos
        to_y, to_x = to_pos
        return 14 if to_y - from_y and to_x - from_x else 10

    def heuristic(self, pos):
        y, x = pos
        goal_y, goal_x = self.goal_pos
        dx, dy = abs(goal_x - x), abs(goal_y - y)
        d_min = min(dx, dy)
        d_max = max(dx, dy)
        return 14 * d_min + 10 * (d_max - d_min)

    def debug(self, nodes):
        self.nodes = nodes

    def astar(self, start_pos, neighbors, goal, start_g, cost, heuristic, 
              limit=sys.maxsize,debug=None):
        """Find the shortest path from start to goal.

        Arguments:

        start_pos      - The starting position.
        neighbors(pos) - A function returning all neighbor positions of the given
                        position.
        goal(pos)      - A function returning true given a goal position, false
                        otherwise.
        start_g        - The starting cost.
        cost(a, b)     - A function returning the cost for moving from one
                        position to another.
        heuristic(pos) - A function returning an estimate of the total cost
                        remaining for reaching goal from the given position.
                        Overestimates can yield suboptimal paths.
        limit          - The maximum number of positions to search.
        debug(nodes)   - This function will be called with a dictionary of all
                        nodes.

        The function returns the best path found. The returned path excludes the
        starting position.
        """

        F, H, NUM, G, POS, OPEN, VALID, PARENT = Planner.F, Planner.H, Planner.NUM, \
                                                  Planner.G, Planner.POS, Planner.OPEN, \
                                                  Planner.VALID, Planner.PARENT

        # Create the start node.
        nums = itertools.count()
        start_h = heuristic(start_pos)
        start = [start_g + start_h, start_h, next(nums), start_g, start_pos, True,
                True, None]

        # Track all nodes seen so far.
        nodes = {start_pos: start}

        # Maintain a heap of nodes.
        heap = [start]

        # Track the best path found so far.
        best = start

        while heap:

            # Pop the next node from the heap.
            current = heappop(heap)
            current[OPEN] = False

            # Have we reached the goal?
            if goal(current[POS]):
                best = current
                break

            # Visit the neighbors of the current node.
            for neighbor_pos in neighbors(current[POS]):
                neighbor_g = current[G] + cost(current[POS], neighbor_pos)
                neighbor = nodes.get(neighbor_pos)
                if neighbor is None:

                    # Limit the search.
                    if len(nodes) >= limit:
                        continue

                    # We have found a new node.
                    neighbor_h = heuristic(neighbor_pos)
                    neighbor = [neighbor_g + neighbor_h, neighbor_h, next(nums),
                                neighbor_g, neighbor_pos, True, True, current[POS]]
                    nodes[neighbor_pos] = neighbor
                    heappush(heap, neighbor)
                    if neighbor_h < best[H]:

                        # We are approaching the goal.
                        best = neighbor

                elif neighbor_g < neighbor[G]:

                    # We have found a better path to the neighbor.
                    if neighbor[OPEN]:

                        # The neighbor is already open. Finding and updating it
                        # in the heap would be a linear complexity operation.
                        # Instead we mark the neighbor as invalid and make an
                        # updated copy of it.

                        neighbor[VALID] = False
                        nodes[neighbor_pos] = neighbor = neighbor[:]
                        neighbor[F] = neighbor_g + neighbor[H]
                        neighbor[NUM] = next(nums)
                        neighbor[G] = neighbor_g
                        neighbor[VALID] = True
                        neighbor[PARENT] = current[POS]
                        heappush(heap, neighbor)

                    else:

                        # Reopen the neighbor.
                        neighbor[F] = neighbor_g + neighbor[H]
                        neighbor[G] = neighbor_g
                        neighbor[PARENT] = current[POS]
                        neighbor[OPEN] = True
                        heappush(heap, neighbor)

            # Discard leading invalid nodes from the heap.
            while heap and not heap[0][VALID]:
                heappop(heap)

        if debug is not None:
            # Pass the dictionary of nodes to the caller.
            debug(nodes)

        # Return the best path as a list.
        path = []
        current = best
        while current[PARENT] is not None:
            path.append(current[POS])
            current = nodes[current[PARENT]]
        path.reverse()
        return path
    
if __name__ == "__main__":
    pass