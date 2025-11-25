# 导入必要的库
import heapq # heapq用于实现优先队列
import math # math提供数学计算功能
import numpy as np # numpy用于数组操作
import matplotlib.pyplot as plt # matplotlib.pyplot用于绘图
 
# 定义Node类：这个类用于表示图中的节点，包含了节点的位置、节点的累计代价值和父节点的索引值信息。
class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x  # x position of node 节点的x 坐标
        self.y = y  # y position of node 节点的y 坐标
        self.cost = cost  # g cost of node 节点的累计代价
        self.pind = pind  # parent index of node 父节点的索引
 
# 定义Para类：这个类用于存储路径规划过程中所需的参数，如环境地图的范围、网格分辨率和运动集合。
class Para:
    def __init__(self, minx, miny, maxx, maxy, xw, yw, reso, motion):
        self.minx = minx #环境中 x 坐标的最小值
        self.miny = miny #环境中 y 坐标的最小值
        self.maxx = maxx #环境中 x 坐标的最大值
        self.maxy = maxy #环境中 y 坐标的最大值
        self.xw = xw #环境的宽度
        self.yw = yw #环境的高度
        self.reso = reso  # resolution of grid world ；网格的分辨率，即每个网格的大小。这会影响地图上障碍物的表示和搜索的精度
        self.motion = motion  # motion set；运动集合，它是一个列表，包含了从当前节点移动到周围节点的不同运动方向
 
 
# astar_planning函数；是A * 算法的主要实现函数，其输入参数包括起点和目标点的x轴、y轴坐标sx、sy、gx、gy，障碍物的x轴、y轴坐标ox、oy ，栅格地图的精度resolution、机器人的半径rr，函数的返回值是规划的路径path
def astar_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    return path of A*.
    :param sx: starting node x [m]
    :param sy: starting node y [m]
    :param gx: goal node x [m]
    :param gy: goal node y [m]
    :param ox: obstacles x positions [m]
    :param oy: obstacles y positions [m]
    :param reso: xy grid resolution
    :param rr: robot radius
    :return: path
    """
 
    #astar_planning函数的第一部分
    # 初始化起点和目标节点
    n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1) # 创建并初始化起点节点 n_start 和目标节点 n_goal，将坐标函数的输入参数中的起点和终点坐标sx, sy...
    n_goal = Node(round(gx / reso), round(gy / reso), 0.0, -1) # gx, gy 除以分辨率reso进行栅格化，并将起点和目标点的累计代价初始化为0、父节点初始化为-1
 
    # 将障碍物坐标按照分辨率进行缩放/栅格化，以适应栅格地图
    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]
  
    # 计算路径规划所需的参数
    P, obsmap = calc_parameters(ox, oy, rr, reso) # 调用 calc_parameters 函数，计算路径规划所需的参数 P 和障碍物地图 obsmap
 
     # 初始化开放和关闭节点集合，以及优先队列
    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_start, P)] = n_start # 初始化A*算法的开放集合 open_set 和闭合集合 closed_set，并将起点节点加入开放集合
 
    q_priority = [] # 初始化一个优先队列 q_priority，其中包含了起点节点的 f 值和索引
    heapq.heappush(q_priority,
                   (fvalue(n_start, n_goal), calc_index(n_start, P)))
 
   #  astar_planning函数的第二部分
   #  A*算法主循环：在每次循环中，算法从开放集合中弹出一个具有最小 f 值的节点，将其加入闭合集合，并探索该节点周围的节点。
   # 对于每个可能的运动方向，都生成一个新的节点，并计算新节点的代价。然后检查节点的合法性，如果节点合法，就根据其在开放集合和闭合集合中的状态进行更新或添加
    while True:
        if not open_set:
            break
 
        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)
 
        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)
 
            if not check_node(node, P, obsmap):
                continue
 
            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority,
                                   (fvalue(node, n_goal), calc_index(node, P)))
 
    # 从关闭节点集合中提取路径
    pathx, pathy = extract_path(closed_set, n_start, n_goal, P) # 通过调用 extract_path 函数，从目标节点开始，回溯父节点的索引，得到从起点到目标的路径
 
    return pathx, pathy # 返回路径的 x 和 y 坐标
 
 
# calc_holonomic_heuristic_with_obstacle函数的输入参数中包含目标点，但不包含起始点
def calc_holonomic_heuristic_with_obstacle(node, ox, oy, reso, rr):
    n_goal = Node(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1) # 将目标节点的坐标按照分辨率进行缩放
 
    # 将障碍物坐标按照分辨率进行缩放
    ox = [x / reso for x in ox]
    oy = [y / reso for y in oy]
 
    # 计算路径规划所需的参数
    P, obsmap = calc_parameters(ox, oy, reso, rr)
 
    # 初始化开放和关闭节点集合，以及优先队列
    open_set, closed_set = dict(), dict()
    open_set[calc_index(n_goal, P)] = n_goal
 
    q_priority = []
    heapq.heappush(q_priority, (n_goal.cost, calc_index(n_goal, P)))
 
    # A*算法主循环
    while True:
        if not open_set:
            break
 
        _, ind = heapq.heappop(q_priority)
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)
 
        for i in range(len(P.motion)):
            node = Node(n_curr.x + P.motion[i][0],
                        n_curr.y + P.motion[i][1],
                        n_curr.cost + u_cost(P.motion[i]), ind)
 
            if not check_node(node, P, obsmap):
                continue
 
            n_ind = calc_index(node, P)
            if n_ind not in closed_set:
                if n_ind in open_set:
                    if open_set[n_ind].cost > node.cost:
                        open_set[n_ind].cost = node.cost
                        open_set[n_ind].pind = ind
                else:
                    open_set[n_ind] = node
                    heapq.heappush(q_priority, (node.cost, calc_index(node, P)))
 
    # 创建启发式地图，并在关闭节点集合中填充代价信息
    hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]
 
    for n in closed_set.values():
        hmap[n.x - P.minx][n.y - P.miny] = n.cost
 
    return hmap
 
# check_node 函数；这个函数主要用于检查给定节点的合法性，即判断节点是否在环境边界内并且不位于障碍物上
def check_node(node, P, obsmap):
    # 检查节点是否超出环境边界，与环境的边界进行比较，如果节点的 x 坐标小于最小 x 坐标 P.minx，
    # 或者大于最大 x 坐标 P.maxx，或者节点的 y 坐标小于最小 y 坐标 P.miny，或者大于最大 y 坐标 P.maxy，那么这个节点就不在合法的环境范围内，应返回 False
    if node.x <= P.minx or node.x >= P.maxx or \
            node.y <= P.miny or node.y >= P.maxy: 
        return False
 
    # 检查节点是否位于障碍物上
    if obsmap[node.x - P.minx][node.y - P.miny]: # 通过检查障碍物地图 obsmap 中与节点位置对应的元素，判断节点是否位于障碍物上
        return False # 如果障碍物地图中对应的位置为 True，表示这个节点处于障碍物上，因此应返回 False
 
    # 如果节点既没有超出边界也不位于障碍物上，返回 True；表示节点可以被用于路径规划
    return True
 
 
def u_cost(u):
    return math.hypot(u[0], u[1])
 
 
def fvalue(node, n_goal):
    return node.cost + h(node, n_goal)
 
 
def h(node, n_goal):
    return math.hypot(node.x - n_goal.x, node.y - n_goal.y)
 
 
def calc_index(node, P):
    return (node.y - P.miny) * P.xw + (node.x - P.minx)
 
 
# calc_parameters函数：实现了路径规划所需的参数的计算，以及相关的数据初始化，并完成了障碍物环境地图的构建
def calc_parameters(ox, oy, rr, reso):
    # 计算环境的边界；使用 min() 和 max() 函数来计算障碍物坐标列表 ox 和 oy 中的最小和最大值，以便确定环境的边界框
    minx, miny = round(min(ox)), round(min(oy))
    maxx, maxy = round(max(ox)), round(max(oy))
    # 计算环境的宽度和高度；通过最小和最大坐标计算出环境的宽度 xw 和高度 yw。这将用于计算地图的网格数量
    xw, yw = maxx - minx, maxy - miny
 
    # 获取运动方向合集
    motion = get_motion()
    # 创建 Para 对象，存储路径规划所需的参数
    P = Para(minx, miny, maxx, maxy, xw, yw, reso, motion)
     # 计算障碍物地图
    obsmap = calc_obsmap(ox, oy, rr, P) # 调用 calc_obsmap() 函数，计算障碍物地图，以便在路径规划中使用 
 
    return P, obsmap # 将 Para 对象 P 和障碍物地图 obsmap 作为元组返回。这些参数将在主要的 A* 路径规划过程中使用，以确保路径规划在正确的环境和条件下进行
 
 
# calc_obsmap()函数实现了障碍物地图的计算，用于在路径规划中表示障碍物的分布情况
def calc_obsmap(ox, oy, rr, P):
    obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)] # 创建一个大小为 P.xw 行、P.yw 列的二维列表 obsmap，用于表示每个网格单元是否是障碍物。初始化所有单元为 False，即默认情况下没有障碍物
 
    for x in range(P.xw): # 通过两个嵌套的循环遍历每个网格单元，以及所有输入的障碍物坐标
        xx = x + P.minx
        for y in range(P.yw):
            yy = y + P.miny
            for oxx, oyy in zip(ox, oy):
                if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso: #内部循环中 使用 math.hypot() 函数计算当前网格单元的中心与每个障碍物之间的距离
                    obsmap[x][y] = True # 如果距离小于等于机器人半径 rr，则说明该网格单元与障碍物相交，因此将该网格标记为 True，表示存在障碍物。
                    break # 随后，通过 break 退出内部循环，因为已经确认这个网格单元是一个障碍物
 
    return obsmap # 函数返回表示障碍物地图的二维布尔列表 obsmap，其中每个元素代表一个网格单元是否有障碍物。这个障碍物地图将在路径规划过程中用于检查每个节点的合法性，以确保规划的路径不会穿过障碍物
 
 
# extract_path；这个函数用于从已经完成路径搜索的关闭节点集合中提取生成的路径；
def extract_path(closed_set, n_start, n_goal, P): # closed_set: 一个字典，包含已经搜索过的节点。键是节点的索引，值是节点对象。n_start: 起始节点。n_goal: 目标节点。P: Para 类的对象，包含了网格世界的参数。
    pathx, pathy = [n_goal.x], [n_goal.y] # 初始化路径的 x 和 y 坐标列表，将目标节点的坐标添加为路径的起点，并获取目标节点的索引
    n_ind = calc_index(n_goal, P)
 
    # 循环根据节点的索引从关闭节点集合中提取路径。循环中的每一步，它将当前节点的坐标添加到路径的 x 和 y 列表中
    while True:
        node = closed_set[n_ind]
        pathx.append(node.x)
        pathy.append(node.y)
        n_ind = node.pind # 它将当前节点的父节点索引（node.pind）作为下一个要提取的节点索引
 
        if node == n_start:
            break # 循环会一直进行，直到当前节点达到起始节点 n_start，此时路径提取完成
 
    pathx = [x * P.reso for x in reversed(pathx)] # 将坐标按照分辨率 P.reso 进行转换，以便将网格世界中的节点坐标转换为实际的环境坐标
    pathy = [y * P.reso for y in reversed(pathy)]
 
    return pathx, pathy # 函数返回路径的 x 和 y 坐标列表
 
# get_motion()函数中编写了运动方向集合，本程序采用八邻域搜索方式，如下所示
def get_motion():
    motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
              [1, 0], [1, -1], [0, -1], [-1, -1]]
 
    return motion
 
# get_env()函数用于定义环境中的障碍物信息，本程序构建的是类似于下图所示的由线条构成的环境
def get_env():
    ox, oy = [], []
 
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)
 
    return ox, oy
 
 
# 主函数 main()，用于调用路径规划算法并绘制结果
def main():
    sx = 10.0  # 起始点的 x 坐标 [m]
    sy = 10.0  # 起始点的 y 坐标 [m]
    gx = 50.0  # 目标点的 x 坐标 [m]
    gy = 50.0  # 目标点的 y 坐标 [m]
 
    robot_radius = 2.0 # 机器人的半径 [m]
    grid_resolution = 1.0  # 网格的分辨率 [m]
    ox, oy = get_env()  # 获取环境中的障碍物坐标
 
    # 使用A*算法进行路径规划
    pathx, pathy = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)
 
    # 绘制环境、路径、起始点和目标点
    plt.plot(ox, oy, 'sk')  # 绘制障碍物
    plt.plot(pathx, pathy, '-r')  # 绘制路径
    plt.plot(sx, sy, 'sg')  # 绘制起始点
    plt.plot(gx, gy, 'sb')  # 绘制目标点
    plt.axis("equal")  # 设置坐标轴比例相等
    plt.show()  # 显示绘图结果
 
 
if __name__ == '__main__':
    main()