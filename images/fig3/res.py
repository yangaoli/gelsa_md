import matplotlib.pyplot as plt
import numpy as np

# 数据准备
GeLSA_cpu = [138, 191, 244, 298, 351, 404, 458, 511, 565, 618]
GeLSA_gpu = [227, 279, 332, 386, 438, 493, 546, 600, 653, 706]
eLSA = [19.4, 50.112, 97.92, 156, 233, 325, 432, 553, 691, 844]
x_1 = np.arange(1000, 11000, 1000)  # 3000-10000，步长1000

# 创建画布
plt.figure(figsize=(6, 6), dpi=100)
plt.grid(True, linestyle='--', alpha=0.6)

# 绘制三条曲线
plt.plot(x_1, GeLSA_cpu, marker='o', label='GeLSA_CPU', linewidth=2, color='#1f77b4')
plt.plot(x_1, GeLSA_gpu, marker='s', label='GeLSA_GPU', linewidth=2, color='#ff7f0e')
plt.plot(x_1, eLSA, marker='^', label='eLSA', linewidth=2, color='#2ca02c')

# 添加标签和标题
plt.xlabel('Sequence Length(n)')
plt.ylabel('Running memory (MB)')
plt.title('Running memory: GeLSA vs eLSA')

# 优化显示
plt.xticks(x_1)  # 确保x轴刻度与数据点对齐
plt.legend()
plt.tight_layout()

# 显示图形
plt.show()