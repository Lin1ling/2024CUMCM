import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm
import matplotlib.font_manager as fm

def calculate_size(E, Z, p):
    """
    计算单侧检验所需的样本量 n
    
    参数:
    E -- 允许误差 
    Z -- 单侧置信度对应的 Z 值 (如 1.645 对应 95% 单侧置信度)
    p -- 标称值 
    
    返回:
    所需的样本量 n (向上取整)
    """
    n = (pow(Z,2) * p * (1 - p)) / pow(E,2)
    return np.ceil(n)  # 取整

Z1 = 1.645  # 对应95%单侧置信度的Z值(上侧单侧区间，右尾检验)
Z2 = 1.28  # 对应90%单侧置信度的Z值（下侧单侧区间，左尾检验）
p = 0.1  # 给出的标称值为10%

# 生成 允许估计误差E 值的数组
E = np.linspace(0.001, 0.2, 400)

y1 = calculate_size(E, Z1, p)
y2 = calculate_size(E, Z2, p)

# 添加字体路径
font_path = "2024B/SimHei.ttf"  # 替换为实际的字体文件路径
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 绘制函数图
plt.figure(figsize=(8, 6))  # 设置图像大小
plt.plot(E, y1)  # 绘制函数图
plt.xlabel('E')  # x轴标签
plt.ylabel('n')  # y轴标签
plt.title('不同估计误差 E 在信度为95%情况下抽样次数')  # 图像标题
plt.grid(True)  # 显示网格

# 保存图像到文件
plt.savefig('2024B/figure/sen_95.pdf')  # 可以保存为.png, .pdf, .svg等格式

# 绘制函数图
plt.figure(figsize=(8, 6))  # 设置图像大小
plt.plot(E, y2)  # 绘制函数图
plt.xlabel('E')  # x轴标签
plt.ylabel('n')  # y轴标签
plt.title('不同估计误差 E 在信度为90%情况下抽样次数')  # 图像标题
plt.grid(True)  # 显示网格

# 保存图像到文件
plt.savefig('sen_90.pdf')  # 可以保存为.png, .pdf, .svg等格式
