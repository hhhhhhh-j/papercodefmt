import numpy as np
from matplotlib.patches import Polygon

def draw_agent(ax, x, y, yaw, color="blue"):
    """
    绘制矩形小车（改进版，解决被覆盖问题）
    """

    L = 4.0   # 车长
    W = 2.0   # 车宽

    # 定义矩形（局部坐标）
    pts = np.array([
        [-L/2, -W/2],   # 左后
        [-L/2,  W/2],   # 右后
        [ L/2,  W/2],   # 右前
        [ L/2, -W/2],   # 左前
    ])

    # 旋转矩阵
    R = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)],
    ])

    # 坐标变换
    pts_world = (R @ pts.T).T
    pts_world[:, 0] += x
    pts_world[:, 1] += y

    # 关键：zorder=10，让车一定显示在图层最上面
    polygon = Polygon(pts_world, closed=True, color=color, zorder=10)

    ax.add_patch(polygon)
