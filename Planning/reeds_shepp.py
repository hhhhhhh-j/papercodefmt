# 导入相关库
import time
import math
import numpy as np
 
 
# parameters initiation 初始化参数
STEP_SIZE = 0.2      # 步长
MAX_LENGTH = 1000.0  # 最大长度
PI = math.pi         # π
 
 
# class for PATH element 定义PATH类
class PATH:
    def __init__(self, lengths, ctypes, L, x, y, yaw, directions):
        self.lengths = lengths              # 包含了路径各部分长度的列表 (+: forward, -: backward) [float]
        self.ctypes = ctypes                # 包含了路径各部分类型的列表 [string]
        self.L = L                          # 整个路径的总长度 [float]
        self.x = x                          # 表示路径最终位置的x坐标 [m]
        self.y = y                          # 表示路径最终位置的y坐标 [m]
        self.yaw = yaw                      # 表示路径最终位置的偏航角（偏转角） [rad]
        self.directions = directions        # 包含了路径各部分方向的列表。列表中的每个元素都是一个整数，表示路径的每个部分的方向。正数表示正向前进，负数表示反向后退  forward: 1, backward:-1
 
 
#  calc_optimal_path函数，用于计算最优路径
def calc_optimal_path(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE):     # 起始位置、起始偏航角、目标位置、目标偏航角、最大曲率和步长大小
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=step_size) # 调用calc_all_paths函数来计算所有可能的路径
 
    minL = paths[0].L    # 首先将第一个路径的总长度设为当前的最小长度，并将对应的索引设为0
    mini = 0
 
    for i in range(len(paths)):    # 对于每个路径，它将比较路径的总长度与当前最小长度的大小
        if paths[i].L <= minL:     
            minL, mini = paths[i].L, i   # 如果路径的总长度小于等于当前最小长度，则将路径的总长度更新为当前最小长度，并将对应的索引更新为当前路径的索引
 
    return paths[mini]    # 函数返回总长度最小的路径对象
 
 
#  calc_all_paths函数，用于计算所有可能的路径
def calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=STEP_SIZE): # 该函数的输入参数包括起始位置、起始偏航角、目标位置、目标偏航角、最大曲率和步长大小（默认为STEP_SIZE）
    q0 = [sx, sy, syaw]     # 起始位置及其对应偏航角保存在变量q0中
    q1 = [gx, gy, gyaw]     # 目标位置以及其对应的偏航角保存在变量q1中
 
    paths = generate_path(q0, q1, maxc)  # 调用generate_path函数生成路径；起始位置、目标位置和最大曲率作为参数；返回一个路径对象列表，其中包含了所有可能的路径
 
    for path in paths:
        x, y, yaw, directions = \
            generate_local_course(path.L, path.lengths,                 # 调用generate_local_course函数生成局部路径
                                  path.ctypes, maxc, step_size * maxc)  # 路径的总长度、每个部分路径的长度、每个部分路径的类型、最大曲率和步长作为参数；返回局部路径的x坐标、y坐标、偏航角和方向信息
 
        # 函数将局部路径转换为全局坐标系
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0] for (ix, iy) in zip(x, y)]  # 使用起始位置的偏航角将局部路径的x和y坐标进行旋转和平移
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1] for (ix, iy) in zip(x, y)] # 从而得到全局路径的x和y坐标
        path.yaw = [pi_2_pi(iyaw + q0[2]) for iyaw in yaw]  # 使用起始位置的偏航角将局部路径的偏航角进行旋转，得到全局路径的偏航角
        path.directions = directions                        # 方向信息
        path.lengths = [l / maxc for l in path.lengths]     # 部分路径长度
        path.L = path.L / maxc                              # 总路径长度； 每段路径的长度和总长度被归一化为最大曲率，以便进行比较和分析
 
    return paths      #返回包含所有可能路径的列表
 
 
# set_path函数用于将一个路径添加到路径列表中
def set_path(paths, lengths, ctypes):   # 该函数接受三个参数：paths表示路径列表，lengths表示路径的长度列表，ctypes表示路径的类型列表
    path = PATH([], [], 0.0, [], [], [], [])   # 函数创建一个空的PATH对象
    path.ctypes = ctypes     # 路径的类型和长度分配给该对象的ctypes和lengths属性
    path.lengths = lengths
 
    # 函数遍历已有的路径列表paths，检查是否存在相同类型的路径
    for path_e in paths:
        if path_e.ctypes == path.ctypes:  # 类型相同
            if sum([x - y for x, y in zip(path_e.lengths, path.lengths)]) <= 0.01:  # 两者长度之差小于等于0.01，则直接返回原始路径列表
                return paths  # 不插入新路径
 
    path.L = sum([abs(i) for i in lengths])   # 计算路径的总长度，将其赋值给路径对象的L属性
 
    if path.L >= MAX_LENGTH:  # 路径的总长度超过了最大长度（MAX_LENGTH）
        return paths    # 直接返回原始路径列表，不插入新路径
 
    assert path.L >= 0.01  # 对路径的总长度进行断言，确保其大于等于0.01
    paths.append(path)     # 将路径对象添加到路径列表中
 
    return paths           # 返回更新后的路径列表
 
 
# 48种reeds_sheep曲线计算函数
def LSL(x, y, phi):   # 坐标、角度参数
    u, t = R(x - math.sin(phi), y - 1.0 + math.cos(phi))   # 使用 R 函数计算 u 和 t 的值，R 函数的输入是 x - math.sin(phi) 和 y - 1.0 + math.cos(phi)
 
    if t >= 0.0:
        v = M(phi - t)  # 使用 M 函数计算 v 的值
        if v >= 0.0:
            return True, t, u, v
 
    return False, 0.0, 0.0, 0.0
 
 
def LSR(x, y, phi):
    u1, t1 = R(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2  # u1 的平方赋值给 u1
 
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = M(t1 + theta)
        v = M(t - phi)
 
        if t >= 0.0 and v >= 0.0:
            return True, t, u, v
 
    return False, 0.0, 0.0, 0.0
 
 
def LRL(x, y, phi):
    u1, t1 = R(x - math.sin(phi), y - 1.0 + math.cos(phi))
 
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = M(t1 + 0.5 * u + PI)
        v = M(phi - t + u)
 
        if t >= 0.0 and u <= 0.0:
            return True, t, u, v
 
    return False, 0.0, 0.0, 0.0
 
 
def SCS(x, y, phi, paths):
    flag, t, u, v = SLS(x, y, phi)
 
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "WB", "S"])
 
    flag, t, u, v = SLS(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["S", "R", "S"])
 
    return paths
 
 
def SLS(x, y, phi):
    phi = M(phi)
 
    if y > 0.0 and 0.0 < phi < PI * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v
    elif y < 0.0 and 0.0 < phi < PI * 0.99:
        xd = -y / math.tan(phi) + x
        t = xd - math.tan(phi / 2.0)
        u = phi
        v = -math.sqrt((x - xd) ** 2 + y ** 2) - math.tan(phi / 2.0)
        return True, t, u, v
 
    return False, 0.0, 0.0, 0.0
 
 
def CSC(x, y, phi, paths):
    flag, t, u, v = LSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["WB", "S", "WB"])
 
    flag, t, u, v = LSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["WB", "S", "WB"])
 
    flag, t, u, v = LSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "R"])
 
    flag, t, u, v = LSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "R"])
 
    flag, t, u, v = LSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["WB", "S", "R"])
 
    flag, t, u, v = LSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["WB", "S", "R"])
 
    flag, t, u, v = LSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "S", "WB"])
 
    flag, t, u, v = LSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "S", "WB"])
 
    return paths
 
 
def CCC(x, y, phi, paths):
    flag, t, u, v = LRL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["WB", "R", "WB"])
 
    flag, t, u, v = LRL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["WB", "R", "WB"])
 
    flag, t, u, v = LRL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, v], ["R", "WB", "R"])
 
    flag, t, u, v = LRL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -v], ["R", "WB", "R"])
 
    # backwards
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
 
    flag, t, u, v = LRL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["WB", "R", "WB"])
 
    flag, t, u, v = LRL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["WB", "R", "WB"])
 
    flag, t, u, v = LRL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, t], ["R", "WB", "R"])
 
    flag, t, u, v = LRL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, -t], ["R", "WB", "R"])
 
    return paths
 
 
def calc_tauOmega(u, v, xi, eta, phi):
    delta = M(u - v)
    A = math.sin(u) - math.sin(delta)
    B = math.cos(u) - math.cos(delta) - 1.0
 
    t1 = math.atan2(eta * A - xi * B, xi * A + eta * B)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
 
    if t2 < 0:
        tau = M(t1 + PI)
    else:
        tau = M(t1)
 
    omega = M(tau - u + v - phi)
 
    return tau, omega
 
 
def LRLRn(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.sqrt(xi * xi + eta * eta))
 
    if rho <= 1.0:
        u = math.acos(rho)
        t, v = calc_tauOmega(u, -u, xi, eta, phi)
        if t >= 0.0 and v <= 0.0:
            return True, t, u, v
 
    return False, 0.0, 0.0, 0.0
 
 
def LRLRp(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
 
    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -0.5 * PI:
            t, v = calc_tauOmega(u, u, xi, eta, phi)
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v
 
    return False, 0.0, 0.0, 0.0
 
 
def CCCC(x, y, phi, paths):
    flag, t, u, v = LRLRn(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["WB", "R", "WB", "R"])
 
    flag, t, u, v = LRLRn(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["WB", "R", "WB", "R"])
 
    flag, t, u, v = LRLRn(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, -u, v], ["R", "WB", "R", "WB"])
 
    flag, t, u, v = LRLRn(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, u, -v], ["R", "WB", "R", "WB"])
 
    flag, t, u, v = LRLRp(x, y, phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["WB", "R", "WB", "R"])
 
    flag, t, u, v = LRLRp(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["WB", "R", "WB", "R"])
 
    flag, t, u, v = LRLRp(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, u, u, v], ["R", "WB", "R", "WB"])
 
    flag, t, u, v = LRLRp(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, -u, -u, -v], ["R", "WB", "R", "WB"])
 
    return paths
 
 
def LRSR(x, y, phi):
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(-eta, xi)
 
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = M(t + 0.5 * PI - phi)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v
 
    return False, 0.0, 0.0, 0.0
 
 
def LRSL(x, y, phi):
    xi = x - math.sin(phi)
    eta = y - 1.0 + math.cos(phi)
    rho, theta = R(xi, eta)
 
    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = M(theta + math.atan2(r, -2.0))
        v = M(phi - 0.5 * PI - t)
        if t >= 0.0 and u <= 0.0 and v <= 0.0:
            return True, t, u, v
 
    return False, 0.0, 0.0, 0.0
 
 
def CCSC(x, y, phi, paths):
    flag, t, u, v = LRSL(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["WB", "R", "S", "WB"])
 
    flag, t, u, v = LRSL(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["WB", "R", "S", "WB"])
 
    flag, t, u, v = LRSL(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["R", "WB", "S", "R"])
 
    flag, t, u, v = LRSL(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "WB", "S", "R"])
 
    flag, t, u, v = LRSR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["WB", "R", "S", "R"])
 
    flag, t, u, v = LRSR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["WB", "R", "S", "R"])
 
    flag, t, u, v = LRSR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, v], ["R", "WB", "S", "WB"])
 
    flag, t, u, v = LRSR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, -v], ["R", "WB", "S", "WB"])
 
    # backwards
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)
 
    flag, t, u, v = LRSL(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["WB", "S", "R", "WB"])
 
    flag, t, u, v = LRSL(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["WB", "S", "R", "WB"])
 
    flag, t, u, v = LRSL(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "WB", "R"])
 
    flag, t, u, v = LRSL(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "WB", "R"])
 
    flag, t, u, v = LRSR(xb, yb, phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["R", "S", "R", "WB"])
 
    flag, t, u, v = LRSR(-xb, yb, -phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["R", "S", "R", "WB"])
 
    flag, t, u, v = LRSR(xb, -yb, -phi)
    if flag:
        paths = set_path(paths, [v, u, -0.5 * PI, t], ["WB", "S", "WB", "R"])
 
    flag, t, u, v = LRSR(-xb, -yb, phi)
    if flag:
        paths = set_path(paths, [-v, -u, 0.5 * PI, -t], ["WB", "S", "WB", "R"])
 
    return paths
 
 
def LRSLR(x, y, phi):
    # formula 8.11 *** TYPO IN PAPER ***
    xi = x + math.sin(phi)
    eta = y - 1.0 - math.cos(phi)
    rho, theta = R(xi, eta)
 
    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= 0.0:
            t = M(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = M(t - phi)
 
            if t >= 0.0 and v >= 0.0:
                return True, t, u, v
 
    return False, 0.0, 0.0, 0.0
 
 
def CCSCC(x, y, phi, paths):
    flag, t, u, v = LRSLR(x, y, phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["WB", "R", "S", "WB", "R"])
 
    flag, t, u, v = LRSLR(-x, y, -phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["WB", "R", "S", "WB", "R"])
 
    flag, t, u, v = LRSLR(x, -y, -phi)
    if flag:
        paths = set_path(paths, [t, -0.5 * PI, u, -0.5 * PI, v], ["R", "WB", "S", "R", "WB"])
 
    flag, t, u, v = LRSLR(-x, -y, phi)
    if flag:
        paths = set_path(paths, [-t, 0.5 * PI, -u, 0.5 * PI, -v], ["R", "WB", "S", "R", "WB"])
 
    return paths
 
 
# generate_local_cours函数用于生成局部路径
def generate_local_course(L, lengths, mode, maxc, step_size): # 总路径的长度L、局部路径长度的列表lengths、局部路径模式的列表mode、 控制曲线的最大曲率maxc、生成路径时进行插值时的步长大小step_size
    point_num = int(L / step_size) + len(lengths) + 3         # 总路径长度和步长大小计算局部路径的点数（point_num）
 
    # 创建空列表 px、py、pyaw 和 directions，用于存储路径点的 x 坐标、y 坐标、偏航角和方向信息
    px = [0.0 for _ in range(point_num)]
    py = [0.0 for _ in range(point_num)]
    pyaw = [0.0 for _ in range(point_num)]
    directions = [0 for _ in range(point_num)]
    ind = 1                             # 初始化变量 ind 为 1，directions[0] 为起始路径的方向（正向或反向）
 
    if lengths[0] > 0.0:     # 根据起始路径的长度，确定路径方向（正向为正步长，反向为负步长）；进入循环，依次处理每个局部路径
        directions[0] = 1
    else:
        directions[0] = -1
 
    if lengths[0] > 0.0:     # 计算局部路径的步长方向 d
        d = step_size
    else:
        d = -step_size
 
    ll = 0.0
 
    for m, l, i in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size
 
        ox, oy, oyaw = px[ind], py[ind], pyaw[ind]   # 获取前一个路径点的坐标和偏航角作为起点。
 
        ind -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:  # 前一个路径点的索引，以及前一个路径长度和当前路径长度的符号关系，计算当前路径段的步长方向 pd；在当前路径段的长度范围内，以步长 d 进
            pd = -d - ll
        else:
            pd = d - ll
 
        while abs(pd) <= abs(l):  # 行插值，生成路径点，并更新路径信息
            ind += 1
            px, py, pyaw, directions = \
                interpolate(ind, pd, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)
            pd += d
 
        ll = l - pd - d  # 计算剩余长度
 
        ind += 1
        px, py, pyaw, directions = \
            interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions)   # 在当前路径段的末尾，以当前路径长度进行插值，生成路径点，并更新路径信息
 
    if len(px) <= 1:
        return [], [], [], []
 
    # remove unused data；移除未使用的数据，即末尾多余的零值点
    while len(px) >= 1 and px[-1] == 0.0:
        px.pop()
        py.pop()
        pyaw.pop()
        directions.pop()
 
    return px, py, pyaw, directions    # 返回生成的路径点列表 px、py、pyaw 和 directions。
 
 
# interpolate函数
def interpolate(ind, l, m, maxc, ox, oy, oyaw, px, py, pyaw, directions):  # 插值函数 interpolate
    if m == "S":  # 直线段（“S”）
        px[ind] = ox + l / maxc * math.cos(oyaw)    # 如果路径段是直线段，则计算直线段上每个点的 x 和 y 坐标，基于前一个路径点的坐标、偏航角和路径长度。
        py[ind] = oy + l / maxc * math.sin(oyaw)    # 更新路径点列表 px、py 和 pyaw。
        pyaw[ind] = oyaw
    else:  # 曲线段
        ldx = math.sin(l) / maxc  # 曲线段在x方向上的增量
        if m == "WB":
            ldy = (1.0 - math.cos(l)) / maxc   # 曲线段在y方向上的增量
        elif m == "R":
            ldy = (1.0 - math.cos(l)) / (-maxc)
 
        gdx = math.cos(-oyaw) * ldx + math.sin(-oyaw) * ldy  # 增量根据前一个路径点的偏航角进行旋转和平移
        gdy = -math.sin(-oyaw) * ldx + math.cos(-oyaw) * ldy # 计算得到当前路径点的坐标
        px[ind] = ox + gdx
        py[ind] = oy + gdy  # 更新路径点列表 px 和 py
 
    if m == "WB":
        pyaw[ind] = oyaw + l  # 计算当前路径点的偏航角，并更新路径点列表 pyaw
    elif m == "R":
        pyaw[ind] = oyaw - l
 
    if l > 0.0:
        directions[ind] = 1 # 确定方向
    else:
        directions[ind] = -1
 
    return px, py, pyaw, directions  # 返回更新后的路径点列表 px、py、pyaw 和 directions
 
 
# generate_path函数，用于生成从起始状态到目标状态的路径
def generate_path(q0, q1, maxc): # q0: 起始状态，包含 x、y 和偏航角（以弧度表示）的列表。q1: 目标状态，与 q0 结构相同； maxc: 控制曲线的最大曲率
    dx = q1[0] - q0[0]
    dy = q1[1] - q0[1]
    dth = q1[2] - q0[2]    # 计算 x、y 和偏航角的差值 dx、dy 和 dth
    c = math.cos(q0[2])
    s = math.sin(q0[2])    # 根据起始状态的偏航角，计算出一个旋转矩阵的元素 c 和 s
    x = (c * dx + s * dy) * maxc
    y = (-s * dx + c * dy) * maxc   # 根据旋转矩阵将差值 dx 和 dy 进行旋转和缩放，得到 x 和 y
 
    paths = []                        # 创建一个空的路径列表 paths
    paths = SCS(x, y, dth, paths)
    paths = CSC(x, y, dth, paths)
    paths = CCC(x, y, dth, paths)
    paths = CCCC(x, y, dth, paths)
    paths = CCSC(x, y, dth, paths)
    paths = CCSCC(x, y, dth, paths)   # 将 x、y 和 dth 分别传递给一系列路径生成函数（SCS、CSC、CCC、CCCC、CCSC 和 CCSCC），并在每次调用后更新 paths
 
    return paths                      # 返回生成的路径列表 paths。
 
 
# pi_2_pi函数用于将给定的角度值 theta 转换到区间 [-π, π] 内
def pi_2_pi(theta):  #循环的目的是将角度值转换为 [-π, π] 范围内的值
    while theta > PI:
        theta -= 2.0 * PI  # 从 theta 中减去 2π，直到 theta 小于或等于 π
 
    while theta < -PI:
        theta += 2.0 * PI  # 将 theta 加上 2π，直到 theta 大于或等于 -π
 
    return theta
 
# R函数用于计算点 (x, y) 的极坐标 (r, theta)
def R(x, y):
    """
    Return the polar coordinates (r, theta) of the point (x, y)
    """
    r = math.hypot(x, y)        # math.hypot(x, y) 函数计算点 (x, y) 到原点的距离 r
    theta = math.atan2(y, x)    # math.atan2(y, x) 函数计算点 (x, y) 的极角 theta
 
    return r, theta             # 极坐标 (r, theta)
 
# M函数用于将给定角度 theta 调整到 -pi <= theta < pi 的范围内
def M(theta):
    """
    Regulate theta to -pi <= theta < pi
    """
    phi = theta % (2.0 * PI)  # 取模运算 % 将输入角度 theta 转换为在 0 到 2*pi 之间的角度 phi
 
    if phi < -PI:
        phi += 2.0 * PI
    if phi > PI:
        phi -= 2.0 * PI
 
    return phi
 
# get_label函数用于根据给定路径 path 生成一个标签。路径是由一系列运动模式（m）和对应的长度（l）组成的
def get_label(path):
    label = ""   # 定义一个空字符串 label，用于存储生成的标签
 
    for m, l in zip(path.ctypes, path.lengths):   # 使用 zip 函数将运动模式和长度两个列表进行逐个配对，依次遍历路径中的每个运动模式和长度
        label = label + m                         # 每个配对的运动模式和长度，将运动模式添加到 label 字符串中
        if l > 0.0:
            label = label + "+"   # 长度 l 大于 0.0，则在运动模式后面添加一个正号 +
        else:
            label = label + "-"   # 否则添加一个负号 -
 
    return label
 
# calc_curvature函数的目的是根据路径的坐标和方向信息计算路径上各点的曲率，并返回曲率和路径段长度的列表
def calc_curvature(x, y, yaw, directions):
    c, ds = [], []                     # 定义两个空列表 c 和 ds，用于存储计算得到的曲率和路径段长度
 
    for i in range(1, len(x) - 1):     # 使用 range 函数从索引1开始，遍历路径中的每个点，直到倒数第2个点为止
        dxn = x[i] - x[i - 1]
        dxp = x[i + 1] - x[i]
        dyn = y[i] - y[i - 1]
        dyp = y[i + 1] - y[i]          # 计算该点与前后相邻点的坐标差值
        dn = math.hypot(dxn, dyn)
        dp = math.hypot(dxp, dyp)      # 计算出前后两个向量的模长
        dx = 1.0 / (dn + dp) * (dp / dn * dxn + dn / dp * dxp)
        ddx = 2.0 / (dn + dp) * (dxp / dp - dxn / dn)
        dy = 1.0 / (dn + dp) * (dp / dn * dyn + dn / dp * dyp)
        ddy = 2.0 / (dn + dp) * (dyp / dp - dyn / dn)
        curvature = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)    # 根据差值和模长计算出局部斜率和曲率，并考虑曲率的正负。如果曲率为非数值（NaN），将其设置为0.0。
        d = (dn + dp) / 2.0
 
        if np.isnan(curvature):
            curvature = 0.0
 
        if directions[i] <= 0.0:
            curvature = -curvature  # 曲率取反
 
        if len(c) == 0:   # 如果 c 列表为空（即第一个点），则将当前路径段长度 d 添加到 ds 列表中，并将当前曲率 curvature 添加到 c 列表中。
            ds.append(d)  
            c.append(curvature)
 
        ds.append(d)      # 对于其他点，将路径段长度 d 添加到 ds 列表中，并将曲率 curvature 添加到 c 列表中
        c.append(curvature)
 
    ds.append(ds[-1])     # 将最后一个路径段的长度和曲率添加到 ds 和 c 列表的末尾，使其长度与输入的 x 和 y 列表相同
    c.append(c[-1])
 
    return c, ds          # 返回计算得到的曲率列表 c 和路径段长度列表 ds
 
# check_path函数用于检查计算得到的路径是否满足一些约束条件。
def check_path(sx, sy, syaw, gx, gy, gyaw, maxc):  # 起点坐标 sx 和 sy、起点偏航角 syaw、终点坐标 gx 和 gy、终点偏航角 gyaw，以及最大曲率 maxc
    paths = calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc)  # 计算从起点到终点的所有可能路径
 
    assert len(paths) >= 1          # 使用 assert 语句确保计算得到的路径列表 paths 不为空（长度至少为1），如果为空，则会触发异常
 
    for path in paths:
        assert abs(path.x[0] - sx) <= 0.01    # 使用 assert 语句确保起点和终点的坐标、偏航角与输入的起点 sx、sy、syaw 以及终点 gx、gy、gyaw 的坐标、偏航角相差不超过0.01。
        assert abs(path.y[0] - sy) <= 0.01    # 如果超过了这个阈值，也会触发异常。
        assert abs(path.yaw[0] - syaw) <= 0.01
        assert abs(path.x[-1] - gx) <= 0.01
        assert abs(path.y[-1] - gy) <= 0.01
        assert abs(path.yaw[-1] - gyaw) <= 0.01
 
        # course distance check检查路径上相邻两点之间的距离是否接近预设的步长 STEP_SIZE
        d = [math.hypot(dx, dy)
             for dx, dy in zip(np.diff(path.x[0:len(path.x) - 1]),
                               np.diff(path.y[0:len(path.y) - 1]))]
 
        for i in range(len(d)):
            assert abs(d[i] - STEP_SIZE) <= 0.001  # 对于每个路径，计算路径上相邻两点的距离 d 并与 STEP_SIZE 比较，如果距离与 STEP_SIZE 的差值超过0.001，则会触发异常。
 
# 主函数中执行路径规划的示例并进行动画展示
def main():
    start_x = 3.0  # 定义了一个状态列表 states，其中包含了一系列起点和终点的坐标以及偏航角。
    start_y = 10.0  # [m]
    start_yaw = np.deg2rad(40.0)  # [rad]
    end_x = 0.0  # [m]
    end_y = 1.0  # [m]
    end_yaw = np.deg2rad(0.0)  # [rad]
    max_curvature = 0.1    # 定义最大曲率 max_c，用于路径规划过程中限制曲率的最大值
 
    t0 = time.time()   # 记录程序开始运行的时间 t0
 
    for i in range(1000):    # 通过循环调用 calc_optimal_path 函数进行路径计算，循环次数为 1000 次
        _ = calc_optimal_path(start_x, start_y, start_yaw, end_x, end_y, end_yaw, max_curvature) # 对于每对相邻状态，提取起点的 x、y 坐标和偏航角，以及终点的 x、y 坐标和偏航角，并调用 calc_optimal_path 函数计算最优路径
 
    t1 = time.time()    # 记录程序结束运行的时间 t1
    print(t1 - t0)      # 打印出程序运行的时间差，即计算 1000 次路径所需的时间
 
 
if __name__ == '__main__':
    main()