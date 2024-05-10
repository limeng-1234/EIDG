import matplotlib.pyplot as plt
import numpy as np
import math

class Vehicle:
    def __init__(self, initial_speed, initial_position):
        self.speed = initial_speed  # 初始速度（m/s）
        self.position = initial_position  # 初始位置（m）
        self.acceleration = 0  # 初始加速度（m/s^2）

    def update(self, time_step, acceleration):
        # 更新车辆状态
        self.acceleration = acceleration
        self.speed += acceleration * time_step
        self.position += self.speed * time_step + 0.5 * acceleration * time_step ** 2





def calculate_acceleration(s, v, delta_v):
    # 定义模型参数
    a_max = 4  # 最大加速度，单位：m/s^2
    v0 = 20   # 期望速度，单位：m/s
    T = 1   # 最小时间头距，单位：s
    s0 = 10    # 最小安全距离，单位：m
    delta = 6 # 速度影响因子
    b = 4    # 舒适减速度，单位：m/s^2

    # 计算s_star
    s_star = s0 + max(0, v * T + (v * delta_v) / (2 * math.sqrt(a_max * b)))

    # 计算加速度
    acceleration = a_max * (1 - (v / v0) ** delta - (s_star / s) ** 2)
    return acceleration


# 创建初始图
fig, ax = plt.subplots()
plt.ion()  # 打开交互模式

# 定义前视范围和后视范围
front_view = 80  # 前视范围，单位：米
rear_view = 80  # 后视范围，单位：米

# 创建一个包含不同s1和s2值的列表
s1_values = [100, 150, 200, 250]
s2_values = [180, 220, 280, 320]

# 循环调用fcn函数
def plot_vehicle(s1, s2):
    ax.clear()  # 清除前一次迭代的图
    # 计算x轴范围
    x_min = s1 - front_view
    x_max = s1 + rear_view

    # 绘制道路
    road_length = front_view + rear_view  # 道路长度，单位：米
    road_width = 3.5  # 道路宽度，单位：米
    ax.add_patch(plt.Rectangle((x_min, 0), road_length, road_width, edgecolor='k', facecolor='w'))

    # 定义车辆的绝对尺寸
    car_length = 4  # 车辆的长度，单位：米
    car_width = 2  # 车辆的宽度，单位：米

    # 绘制第一辆车
    car1_center = s1  # 第一辆车的质心位置，单位：米
    ax.add_patch(plt.Rectangle((car1_center - car_length / 2, (road_width - car_width) / 2), car_length, car_width,
                               edgecolor='r', facecolor='none'))

    # 绘制第二辆车
    car2_center = s2  # 第二辆车的质心位置，单位：米
    ax.add_patch(plt.Rectangle((car2_center - car_length / 2, (road_width - car_width) / 2), car_length, car_width,
                               edgecolor='b', facecolor='none'))

    # 设置坐标轴
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([0, road_width])

    # 添加标签和标题
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Width (m)')
    ax.set_title('Road with Cars')

    plt.pause(0.1)  # 等待一段时间以便查看图像

# 关闭交互模式
plt.ioff()
plt.show()
