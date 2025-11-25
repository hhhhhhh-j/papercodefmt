# 导入库和模块
import os
import sys
import math               # heapq和heapdict用于构建优先级队列；numpy和matplotlib用于数值计算和绘图，scipy.spatial.kdtree用于创建K-D树来进行碰撞检测
import heapq
from heapdict import heapdict
import time
import numpy as np              
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../HybridAstarPlanner/")
 
import astar as astar
import draw as draw
import reeds_shepp as rs
 
# 定义参数配置类 C；用于存储路径规划中使用的各种参数
class C:  # Parameter config
    PI = math.pi
 
    XY_RESO = 2.0                # 坐标分辨率，表示每个坐标点在X和Y方向上的间隔
    YAW_RESO = np.deg2rad(15.0)  # 角度分辨率，表示每个角度的间隔且将角度转换为弧度
    MOVE_STEP = 0.4              # 插值分辨率，表示在路径中每个节点之间的距离
    N_STEER = 20.0               # 计算转向角的离散化值，即将最大转向角范围分成多少个不同的离散值
    COLLISION_CHECK_STEP = 5     # 碰撞检测时，跳过的节点数，即每隔多少个节点进行一次碰撞检测，用于减少碰撞检测的计算量
    EXTEND_BOUND = 1             # 碰撞检测范围的扩展值，用于在实际碰撞检测时，将检测范围扩展一定距离。
 
    GEAR_COST = 100.0            # 切换方向的惩罚成本，当车辆需要切换行驶方向时，会增加这个成本
    BACKWARD_COST = 5.0          # 后退行驶的惩罚成本
    STEER_CHANGE_COST = 5.0      # 转向角变化的惩罚成本
    STEER_ANGLE_COST = 1.0       # 转向角度的惩罚成本
    H_COST = 15.0                # 启发式成本的惩罚成本，用于引导启发式搜索算法的探索方向
 
    RF = 4.5      # 车辆从后部到车辆前部的距离
    RB = 1.0      # 车辆从后部到车辆后部的距离
    W = 3.0       # 车辆的宽度
    WD = 0.7 * W  # 左右车轮之间的距离，用于计算车辆的转向角
    WB = 3.5      # 车辆的轴距，表示前后轮之间的距离
    TR = 0.5      # 车轮的半径
    TW = 1        # 车轮的宽度
    MAX_STEER = 0.6  # 最大转向角，表示车辆的最大转向角度
 
# 定义节点类 Node表示路径规划中的节点
class Node:
    def __init__(self, xind, yind, yawind, direction, x, y, 
                 yaw, directions, steer, cost, pind):       
        self.xind = xind              # xind：节点所在的 X 坐标索引
        self.yind = yind              # yind：节点所在的 Y 坐标索引
        self.yawind = yawind          # 节点的航向角索引
        self.direction = direction    # 车辆的行驶方向，值为 1 表示正向，-1 表示反向；
        self.x = x
        self.y = y                    # 车辆路径中的坐标x,y
        self.yaw = yaw                # 车辆路径中的航向角列表
        self.directions = directions  # 车辆在每个路径节点上的行驶方向列表
        self.steer = steer            # 转向角
        self.cost = cost              # 从起始节点到当前节点的路径代价
        self.pind = pind              # 父节点在路径规划中的索引，用于追踪路径的连接
 
# 存储路径规划所需的各种参数和计算结果
class Para:
    def __init__(self, minx, miny, minyaw, maxx, maxy, maxyaw,
                 xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx             # 地图中 X 坐标的最小值
        self.miny = miny             # 地图中 Y 坐标的最小值
        self.minyaw = minyaw         # 地图中航向角的最小值
        self.maxx = maxx             # 地图中 X 坐标的最大值
        self.maxy = maxy             # 地图中 Y 坐标的最大值
        self.maxyaw = maxyaw         # 地图中航向角的最大值
        self.xw = xw                 # X 方向上的网格数
        self.yw = yw                 # Y 方向上的网格数
        self.yaww = yaww             # 航向角方向上的网格数
        self.xyreso = xyreso         # XY 坐标分辨率
        self.yawreso = yawreso       # 航向角分辨率
        self.ox = ox                 # 障碍物的 X 坐标列表
        self.oy = oy                 # 障碍物的 Y 坐标列表
        self.kdtree = kdtree         # 使用 k-d 树表示的障碍物数据结构，用于快速查找最近的障碍物点
 
# Path 的类，用于存储路径规划的结果;描述路径节点属性
class Path:
    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost
 
# 优先级队列
class QueuePrior:
    def __init__(self):          # 初始化优先级队列的构造函数
        self.queue = heapdict()  # 创建一个 heapdict 数据结构作为队列的实现，该数据结构是基于堆和字典的混合数据结构，允许在 O(1) 时间内执行插入和删除操作
 
    def empty(self):                 # 判断优先级队列是否为空
        return len(self.queue) == 0  # if Q is empty；如果队列中没有元素，返回 True；否则，返回 False
 
    def put(self, item, priority):   # 将元素 item 插入队列，
        self.queue[item] = priority  # 指定它的优先级为 priority
 
    def get(self):
        return self.queue.popitem()[0]  # 具有最小优先级的弹出元素
 
# hybrid_astar_planning函数实现了混合A*算法
def hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, xyreso, yawreso):
    # 将连续坐标转换为离散坐标
    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)
    
    # 创建起始节点和目标节点
    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [1], 0.0, 0.0, -1)
 
    # 创建KD树以加速障碍物查询
    kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)           # 计算规划参数P
 
    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)       # 使用A*算法计算启发式地图
    steer_set, direc_set = calc_motion_set()                                                    # 计算转向和方向的集合
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}                                  # 初始化open_set、closed_set，将起始节点添加到open_set中
 
    qp = QueuePrior()                                                                           # 初始化优先队列qp
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))
 
    # 开始主循环
    while True:
        if not open_set:
            return None      # 无解情况
 
        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)
 
        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, P) # 尝试使用解析扩展更新节点
 
        if update:
            fnode = fpath
            break
 
        # 遍历转向和方向的集合
        for i in range(len(steer_set)):
            # 计算下一个节点
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)
 
            if not node:
                continue
 
            node_ind = calc_index(node, P)
 
            if node_ind in closed_set:
                continue
 
            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node
                    qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
 
    return extract_path(closed_set, fnode, nstart)   # 提取并返回路径
 
# extract_path函数用于从封闭集（closed set）中提取并构造最终的路径
def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, direc = [], [], [], []  # 创建空列表用于存储路径信息
    cost = 0.0  # 初始化路径成本
    node = ngoal  # 从目标节点开始回溯路径
 
    while True:
        rx += node.x[::-1]  # 将当前节点的x坐标列表反向加入rx列表
        ry += node.y[::-1]  # 将当前节点的y坐标列表反向加入ry列表
        ryaw += node.yaw[::-1]  # 将当前节点的航向角列表反向加入ryaw列表
        direc += node.directions[::-1]  # 将当前节点的方向列表反向加入direc列表
        cost += node.cost  # 将当前节点的成本累加到总成本中
 
        if is_same_grid(node, nstart):  # 如果当前节点已经回溯到起始节点
            break  # 停止回溯
 
        node = closed[node.pind]  # 如果当前节点不是起始节点，获取上一节点，继续回溯
 
    rx = rx[::-1]  # 将rx列表反向，得到正确的路径顺序
    ry = ry[::-1]  # 将ry列表反向，得到正确的路径顺序
    ryaw = ryaw[::-1]  # 将ryaw列表反向，得到正确的路径顺序
    direc = direc[::-1]  # 将direc列表反向，得到正确的路径顺序
 
    direc[0] = direc[1]  # 将起始节点的方向设置为与第二个节点的方向相同
    path = Path(rx, ry, ryaw, direc, cost)  # 创建Path对象，表示完整路径
 
    return path  # 返回构造的路径对象
 
 
#  calc_next_node函数实现了计算下一个路径规划节点的功能
def calc_next_node(n_curr, c_id, u, d, P):   # 当前节点、索引、转向角、进行方向、参数对象
    step = C.XY_RESO * 2   # 计算一个步长
 
    nlist = math.ceil(step / C.MOVE_STEP)    # 计算下一个节点数量
    xlist = [n_curr.x[-1] + d * C.MOVE_STEP * math.cos(n_curr.yaw[-1])]            # 初始化列表，储存下一个节点的x坐标、y坐标、航向角；在 d 方向上行驶 C.MOVE_STEP 距离
    ylist = [n_curr.y[-1] + d * C.MOVE_STEP * math.sin(n_curr.yaw[-1])]
    yawlist = [rs.pi_2_pi(n_curr.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))]
    # 不断累加进行插值，计算出剩余的节点的 x、y 坐标和航向角
    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))
    # 最后一个节点的坐标和航向角，将其转换为在地图上的索引坐标
    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)
    # 调用 is_index_ok 函数检查这些索引坐标是否合法，即是否在地图范围内且不与障碍物冲突。如果不合法，说明生成的节点在地图外或与障碍物冲突，此时会返回 None。
    if not is_index_ok(xind, yind, xlist, ylist, yawlist, P):
        return None
 
    cost = 0.0
 
    if d > 0:
        direction = 1
        cost += abs(step)    # 前进成本
    else:
        direction = -1
        cost += abs(step) * C.BACKWARD_COST    # 后退成本
 
    if direction != n_curr.direction:  # 生成的节点的行进方向和父节点的行进方向不一致
        cost += C.GEAR_COST            # 切换车辆行进方向加上切换成本
 
    cost += C.STEER_ANGLE_COST * abs(u)  # 累加转向角度的惩罚成本
    cost += C.STEER_CHANGE_COST * abs(n_curr.steer - u)  # 转向角度的变化成本
    cost = n_curr.cost + cost
 
    directions = [direction for _ in range(len(xlist))]    # 将生成的行进方向信息复制多次，以匹配生成的节点数量，存储在 directions 列表中
 
    node = Node(xind, yind, yawind, direction, xlist, ylist,       # 创建一个新的节点对象 Node
                yawlist, directions, u, cost, c_id)
 
    return node
 
# is_index_ok函数用于判断生成的路径规划节点的索引坐标是否在地图范围内，并且检查这些节点是否与障碍物发生碰撞
def is_index_ok(xind, yind, xlist, ylist, yawlist, P):
    if xind <= P.minx or \
            xind >= P.maxx or \
            yind <= P.miny or \
            yind >= P.maxy:
        return False          # 如果任何一个条件成立，说明节点位于地图边界外，此时返回 False，表示索引不合法
 
    ind = range(0, len(xlist), C.COLLISION_CHECK_STEP)  # 计算一个索引范围 ind；步长是 C.COLLISION_CHECK_STEP
 
    nodex = [xlist[k] for k in ind]     # 选择在 xlist、ylist 和 yawlist 中的哪些值需要进行碰撞检查
    nodey = [ylist[k] for k in ind]     # 将对应索引处的值从 xlist、ylist 和 yawlist 中提取出来，分别存储在 nodex、nodey 和 nodeyaw 列表中
    nodeyaw = [yawlist[k] for k in ind]
 
    if is_collision(nodex, nodey, nodeyaw, P):
        return False # 调用函数 is_collision，将 nodex、nodey 和 nodeyaw 传递给它，以及参数对象 P，用于检查这些节点是否与障碍物发生碰撞。如果发生碰撞，则返回 False，否则返回 True
 
    return True
 
 
def update_node_with_analystic_expantion(n_curr, ngoal, P):   # 当前节点 、目标节点、 以及参数对象 
    path = analystic_expantion(n_curr, ngoal, P)  # 分析路径规划，当前节点到目标节点是否有更优的路径
 
    if not path:
        return False, None # 如果没有找到更优的路径，则返回 False 表示无法更新节点。
 
    # 如果 analystic_expantion 找到了从当前节点到目标节点的可行Reeds shee曲线路径，它会返回一个路径对象 path，然后代码继续执行
    fx = path.x[1:-1]   # 提取出除去起始和终点的路径信息、分别存储在 fx、fy、fyaw 和 fd 列表中
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]
    fd = path.directions[1:-1]
 
    fcost = n_curr.cost + calc_rs_path_cost(path)  # 新节点的成本由当前节点的成本加上从当前节点到目标节点的路径 path 的成本计算得到
    fpind = calc_index(n_curr, P)   # 计算新节点的索引
    fsteer = 0.0
 
    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fd, fsteer, fcost, fpind)                        # 创建一个新的节点对象 fpath
 
    return True, fpath   # 成功更新节点
 
# analystic_expantion函数会尝试调用Reeds_sheep算法查找48种可行曲线
def analystic_expantion(node, ngoal, P):
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]
 
    maxc = math.tan(C.MAX_STEER) / C.WB  # 车辆最大转向角对应的转弯半径
    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.MOVE_STEP)  # 根据 Reeds-Shepp 路径类型计算从起始状态到目标状态 的所有可能路径
 
    if not paths:
        return None   # 无法进行路径扩展
 
    pq = QueuePrior()  # 如果找到了路径，则创建一个优先级队列 pq
    for path in paths:
        pq.put(path, calc_rs_path_cost(path))
 
    while not pq.empty():
        path = pq.get()  # 优先级队列中依次弹出路径
        ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)  # 使用间隔为 C.COLLISION_CHECK_STEP 的索引，从路径中提取出部分点
 
        pathx = [path.x[k] for k in ind]
        pathy = [path.y[k] for k in ind]
        pathyaw = [path.yaw[k] for k in ind]
 
        if not is_collision(pathx, pathy, pathyaw, P):  # 调用 is_collision 函数检查这些点是否与障碍物发生碰撞
            return path
 
    return None
 
#  is_collision函数用于检测车辆是否与障碍物发生碰撞。
def is_collision(x, y, yaw, P):
    for ix, iy, iyaw in zip(x, y, yaw):
        d = 1                        # 扩展碰撞检测范围
        dl = (C.RF - C.RB) / 2.0     # 车辆前后轮之间的距离差的一半
        r = (C.RF + C.RB) / 2.0 + d  # 车辆的半径
 
        cx = ix + dl * math.cos(iyaw)   # 车辆的前轴中心点坐标 (cx, cy)
        cy = iy + dl * math.sin(iyaw)
 
        ids = P.kdtree.query_ball_point([cx, cy], r)  # kd数近邻点查询，以车辆前部中心为中心，以 r 为半径的范围内的所有障碍物点的索引
 
        if not ids:
            continue
 
        for i in ids:
            xo = P.ox[i] - cx  # 车辆前部中心点的相对坐标 (xo, yo)
            yo = P.oy[i] - cy
            dx = xo * math.cos(iyaw) + yo * math.sin(iyaw)   # 相对坐标通过旋转变换（根据车辆航向角）映射到车辆坐标系
            dy = -xo * math.sin(iyaw) + yo * math.cos(iyaw)
 
            if abs(dx) < r and abs(dy) < C.W / 2 + d:
                return True  # 发生碰撞
 
    return False    # 未发生碰撞
 
# calc_rs_path_cost函数用于计算路径的成本，即评估给定路径的优劣程度
def calc_rs_path_cost(rspath):
    cost = 0.0                 # 初始化路径成本为0
 
    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST   # 后退行驶
 
    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:  # 检查相邻的路径段是否具有不同的行驶方向
            cost += C.GEAR_COST    # 增加换挡的代价
 
    for ctype in rspath.ctypes:
        if ctype != "S":    # 不是直线
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)
 
    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]   # 存储路径每个曲线段对应的转向角度
 
    for i in range(nctypes):
        if rspath.ctypes[i] == "R":   # 右转
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[i] == "WB": # 左转
            ulist[i] = C.MAX_STEER
 
    for i in range(nctypes - 1):   # 遍历转向角度列表
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])
 
    return cost
 
# calc_hybrid_cost函数用于计算Hybrid A * 算法中节点的成本
def calc_hybrid_cost(node, hmap, P):    # 节点、启发式地图、包含地图和路径规划参数的对象
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]  # 启发式地图的成本
 
    return cost
 
# calc_motion_set函数用于计算可用的车辆操控动作集合，包括转向角度和方向
def calc_motion_set():
    s = np.arange(C.MAX_STEER / C.N_STEER,       # 生成转向角度值
                  C.MAX_STEER, C.MAX_STEER / C.N_STEER)
 
    steer = list(s) + [0.0] + list(-s)
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer
 
    return steer, direc
 
#  is_same_grid函数用于判断两个节点是否位于相同的栅格（格子）上
def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False   # 不在相同的栅格上
 
    return True
 
# calc_index(node, P)函数用于计算给定节点在一维数组（通常用于表示二维栅格地图或状态空间）中的索引
def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)
 
    return ind
 
#  calc_parameters函数用于计算参数并返回一个包含计算结果的参数对象
def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minx = round(min(ox) / xyreso)
    miny = round(min(oy) / xyreso)
    maxx = round(max(ox) / xyreso)
    maxy = round(max(oy) / xyreso)
 
    xw, yw = maxx - minx, maxy - miny
 
    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw
 
    return Para(minx, miny, minyaw, maxx, maxy, maxyaw,
                xw, yw, yaww, xyreso, yawreso, ox, oy, kdtree)
 
# 用于在一个matplotlib图中绘制一个表示汽车的简化形状，包括车身和车轮
def draw_car(x, y, yaw, steer, color='black'):
    # 定义车辆轮廓
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])
 
    # 定义车轮形状
    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])
    # 复制车轮形状，用于各个轮子
    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()
    # 构建旋转矩阵
    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])
 
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])
    # 根据转向角度旋转前轮
    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)
    # 调整前轮位置
    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2
    
    # 根据车辆朝向旋转前轮和车轮轮廓
    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)
 
    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)
    # 平移车辆和车轮位置
    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])
     # 绘制车辆和车轮
    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    draw.Arrow(x, y, yaw, C.WB * 0.8, color)
 
# design_obstacles函数用于生成障碍物的坐标
def design_obstacles(x, y):   # 地图的宽度和高度
    ox, oy = [], []
    # 添加四条边界：通过循环，将地图的四条边界添加到障碍物坐标中，以防止车辆越过地图边界
    for i in range(x): 
        ox.append(i)
        oy.append(0)
    for i in range(x):
        ox.append(i)
        oy.append(y - 1)
    for i in range(y):
        ox.append(0)
        oy.append(i)
    for i in range(y):
        ox.append(x - 1)
        oy.append(i)
    for i in range(10, 21):   # 例子
        ox.append(i)
        oy.append(15)
    for i in range(15):
        ox.append(20)
        oy.append(i)
    for i in range(15, 30):
        ox.append(30)
        oy.append(i)
    for i in range(16):
        ox.append(40)
        oy.append(i)
 
    return ox, oy
 
# 用于调用Hybrid A*算法来规划车辆的路径，并在matplotlib中绘制车辆的轨迹和障碍物
def main():
    print("start!")
    x, y = 51, 31    # 地图的宽度和高度 
    sx, sy, syaw0 = 10.0, 7.0, np.deg2rad(120.0)   # 起始点的x，y坐标和偏航角
    gx, gy, gyaw0 = 45.0, 20.0, np.deg2rad(90.0)   # 目标点的x，y坐标和偏航角
 
    # 生成障碍物的坐标
    ox, oy = design_obstacles(x, y)
 
    t0 = time.time()
     # 使用Hybrid A*算法进行路径规划
    path = hybrid_astar_planning(sx, sy, syaw0, gx, gy, gyaw0,
                                 ox, oy, C.XY_RESO, C.YAW_RESO)
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
 
    # 在每个时刻绘制车辆的轨迹和障碍物
    for k in range(len(x)):
        plt.cla()         # 清空当前图形 
        plt.plot(ox, oy, "sk")   # 绘制障碍物和规划路径
        plt.plot(x, y, linewidth=1.5, color='r')
 
        # 计算当前时刻的方向和转向角
        if k < len(x) - 2:
            dy = (yaw[k + 1] - yaw[k]) / C.MOVE_STEP
            steer = rs.pi_2_pi(math.atan(-C.WB * dy / direction[k]))
        else:
            steer = 0.0
 
        # 绘制车辆
        draw_car(gx, gy, gyaw0, 0.0, 'dimgray')   # 绘制目标点
        draw_car(x[k], y[k], yaw[k], steer)       # 绘制当前车辆状态
        plt.title("Hybrid A*")
        plt.axis("equal")
        plt.pause(0.0001)
 
    plt.show()
    print("Done!")
 
 
if __name__ == '__main__':
    main()