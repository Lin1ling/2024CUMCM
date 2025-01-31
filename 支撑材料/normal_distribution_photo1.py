import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.font_manager as fm

# 定义均值和标准差
mu = 0  # 均值
sigma = 1  # 标准差

# 创建正态分布曲线的 x 值
x = np.linspace(-4, 4, 1000) #用4表示右尾区域
y = norm.pdf(x, mu, sigma)

# 添加字体路径
font_path = "SimHei.ttf"  # 替换为实际的字体文件路径
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建图形
plt.figure(figsize=(10, 6))
plt.rc('font', family='SimHei')
# 绘制正态分布曲线
plt.plot(x, y, label='标准正态分布', color='blue')

# 设置单侧检验的 Z 值 (95% 单侧检验)
z_value = 1.645
x_fill = np.linspace(z_value, 4, 1000)
y_fill = norm.pdf(x_fill, mu, sigma)

# 填充单侧检验区域
plt.fill_between(x_fill, y_fill, color='orange', alpha=0.5, label=f'右侧检验区域 (Z = {z_value})')

# 标注 Z 值
plt.axvline(x=z_value, color='orange', linestyle='--', label=f'Z = {z_value}')

# 设置标题和标签
plt.title('右尾检验的标准正态分布图')
plt.xlabel('Z 值')
plt.ylabel('概率密度')

# 显示图例
plt.legend()

# 显示图形
plt.grid(True)
plt.savefig('figure/a.pdf')
