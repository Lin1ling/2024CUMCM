import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib.font_manager as fm

# 观测数据
n = 2436  # 样本总数
k = 243.6   # 次品数

# 先验分布参数（无信息先验 Beta(1, 1)）
alpha_prior = 1
beta_prior = 1

# 计算后验分布参数
alpha_post = alpha_prior + k
beta_post = beta_prior + n - k

# 生成后验分布的样本
x = np.linspace(0, 1, 1000)  # 次品率的可能取值范围（0到1之间）
posterior_pdf = beta.pdf(x, alpha_post, beta_post)

# 绘制次品率的后验分布
plt.plot(x, posterior_pdf, label=f'Beta({alpha_post}, {beta_post})', color='blue')
plt.fill_between(x, posterior_pdf, alpha=0.3, color='blue')

# 输出后验分布的均值和 95% 置信区间
mean_posterior = beta.mean(alpha_post, beta_post)
ci_lower, ci_upper = beta.ppf([0.025, 0.975], alpha_post, beta_post)

# 在图中绘制均值和置信区间
plt.axvline(mean_posterior, color='red', linestyle='--', label=f'均值: {mean_posterior:.4f}')
plt.axvline(ci_lower, color='green', linestyle='--', label=f'90% 置信区间下限: {ci_lower:.4f}')
plt.axvline(ci_upper, color='green', linestyle='--', label=f'90% 置信区间上限: {ci_upper:.4f}')

# 添加字体路径
font_path = "SimHei.ttf"  # 替换为实际的字体文件路径
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = [prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置图的标题和标签
plt.title('次品率的后验分布')
plt.xlabel('次品率 (p)')
plt.ylabel('概率密度')

# 显示图例
plt.legend()
plt.grid(True)
plt.savefig('figure/d.pdf')

# 打印均值和 90% 置信区间
print(f"后验分布的均值为: {mean_posterior:.4f}")
print(f"90% 置信区间为: [{ci_lower:.4f}, {ci_upper:.4f}]")
