import matplotlib.pyplot as plt
import numpy as np
import math
PI = np.pi
 
# Arrow 类函数
class Arrow:
    def __init__(self, x, y, theta, L, c):  # 构造函数，接受箭头的起始位置 (x, y)，方向 theta，长度 L 和颜色 c 作为参数
        angle = np.deg2rad(30)
        d = 0.3 * L
        w = 2
 
         # 计算起始位置和结束位置
        x_start = x
        y_start = y                      
        x_end = x + L * np.cos(theta)
        y_end = y + L * np.sin(theta)      
 
        # 计算头部左右两侧点的位置
        theta_hat_L = theta + PI - angle
        theta_hat_R = theta + PI + angle       
 
        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)
 
        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)
 
        # 使用 plt.plot() 函数绘制箭头的线段和头部
        plt.plot([x_start, x_end], [y_start, y_end], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L],
                 [y_hat_start, y_hat_end_L], color=c, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R],
                 [y_hat_start, y_hat_end_R], color=c, linewidth=w)
 
 
class Car:
    def __init__(self, x, y, yaw, w, L):  # 车辆的中心位置 (x, y)，航向角 yaw，宽度 w 和长度 L 作为参数
        theta_B = PI + yaw
 
        xB = x + L / 4 * np.cos(theta_B)
        yB = y + L / 4 * np.sin(theta_B)
 
        theta_BL = theta_B + PI / 2
        theta_BR = theta_B - PI / 2
 
        x_BL = xB + w / 2 * np.cos(theta_BL)        # 左下顶点
        y_BL = yB + w / 2 * np.sin(theta_BL)
        x_BR = xB + w / 2 * np.cos(theta_BR)        # 右下顶点
        y_BR = yB + w / 2 * np.sin(theta_BR)
 
        x_FL = x_BL + L * np.cos(yaw)               # 左前顶点
        y_FL = y_BL + L * np.sin(yaw)
        x_FR = x_BR + L * np.cos(yaw)               # 右前顶点
        y_FR = y_BR + L * np.sin(yaw)
 
        plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                 [y_BL, y_BR, y_FR, y_FL, y_BL],
                 linewidth=1, color='black')        # 使用 plt.plot() 函数绘制车辆的轮廓，以及调用 Arrow 类来绘制车辆朝向的箭头
 
        Arrow(x, y, yaw, L / 2, 'black')
        # plt.axis("equal")
        # plt.show()
 
 
def draw_car(x, y, yaw, steer, C, color='black'):   # 车辆坐标、航向角、转向角度、车辆尺寸参数的对象、颜色；用于绘制车辆的轮廓和车轮
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],                   
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])    # car 定义了车辆的轮廓
 
    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],                 # wheel 定义了车轮的形状
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])
 
    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()
 
    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],     # Rot1 和 Rot2 是旋转矩阵，用于根据车辆的航向角和转向角旋转车辆和车轮
                     [math.sin(yaw), math.cos(yaw)]])
 
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])
 
    # 通过矩阵运算和平移操作计算出车轮和车辆各个顶点的位置
    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)
 
    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2
 
    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)
 
    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)
 
    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])
 
    # 使用 plt.plot() 函数绘制车辆轮廓和车轮。调用 Arrow 类来绘制车辆朝向的箭头。
    plt.plot(car[0, :], car[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    Arrow(x, y, yaw, C.WB * 0.8, color)
 
 
if __name__ == '__main__':
    # Arrow(-1, 2, 60)
    Car(0, 0, 1, 2, 60)