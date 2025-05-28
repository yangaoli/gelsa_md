


# import statistics

# new_design_data_2 = [
#     336330, 1.22686e+06, 2.77131e+06, 5.2022e+06, 8.99619e+06, 1.14117e+07, 1.63425e+07, 
#     1.99866e+07, 2.61709e+07, 3.2601e+07, 4.17899e+07, 4.9307e+07, 5.62417e+07, 
#     7.16404e+07, 7.39658e+07, 8.74493e+07, 9.68632e+07, 1.08865e+08, 1.31373e+08, 
#     1.3262e+08, 1.43879e+08, 1.72808e+08, 1.91541e+08, 1.90894e+08, 2.01113e+08
# ]

# old_design_data_2 = [
#     613866, 2.57858e+06, 5.67244e+06, 1.00037e+07, 1.57224e+07, 2.28275e+07, 3.07486e+07,
#     4.15605e+07, 5.0949e+07, 6.75636e+07, 7.61441e+07, 9.70095e+07, 1.13652e+08, 
#     1.41285e+08, 1.48954e+08, 1.65699e+08, 1.88489e+08, 2.07974e+08, 2.40617e+08,
#     2.58591e+08, 2.8509e+08, 3.24793e+08, 3.39759e+08, 3.82138e+08, 4.07191e+08
# ]

# time_ratios = [old / new for old, new in zip(old_design_data_2, new_design_data_2)]

# average_ratio = sum(time_ratios) / len(time_ratios)

# std = statistics.variance(time_ratios)

# print(time_ratios)
# print(average_ratio)
# print(std)



























# import matplotlib.pyplot as plt
# import numpy as np

# # 第一组数据（基于假设的x值）
# new_design_values_1 = [3.01785e+07, 5.65973e+07, 8.41229e+07, 1.08117e+08, 1.13602e+08, 1.30785e+08, 1.47589e+08, 1.98282e+08, 2.20129e+08, 2.43879e+08]
# old_design_values_1 = [6.25471e+07, 1.36683e+08, 2.39119e+08, 3.42238e+08, 4.68842e+08, 6.52117e+08, 8.9387e+08, 1.12683e+09, 1.41468e+09, 1.76528e+09]
# x_1 = np.arange(100, 1100, 100)

# # 第二组数据（基于实际的索引）
# new_design_data_2 = [
#     336330, 1.22686e+06, 2.77131e+06, 5.2022e+06, 8.99619e+06, 1.14117e+07, 1.63425e+07, 
#     1.99866e+07, 2.61709e+07, 3.2601e+07, 4.17899e+07, 4.9307e+07, 5.62417e+07, 
#     7.16404e+07, 7.39658e+07, 8.74493e+07, 9.68632e+07, 1.08865e+08, 1.31373e+08, 
#     1.3262e+08, 1.43879e+08, 1.72808e+08, 1.91541e+08, 1.90894e+08, 2.01113e+08
# ]
# old_design_data_2 = [
#     613866, 2.57858e+06, 5.67244e+06, 1.00037e+07, 1.57224e+07, 2.28275e+07, 3.07486e+07,
#     4.15605e+07, 5.0949e+07, 6.75636e+07, 7.61441e+07, 9.70095e+07, 1.13652e+08, 
#     1.41285e+08, 1.48954e+08, 1.65699e+08, 1.88489e+08, 2.07974e+08, 2.40617e+08,
#     2.58591e+08, 2.8509e+08, 3.24793e+08, 3.39759e+08, 3.82138e+08, 4.07191e+08
# ]
# x_2 = np.arange(200, 5200, 200)

# # 创建包含两个子图的图表
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# # 绘制第一组数据
# ax1.plot(x_1, new_design_values_1, marker='o', label='New Design Algorithm')
# ax1.plot(x_1, old_design_values_1, marker='x', label='Old Design Algorithm')
# ax1.set_title('Comparison Based on Hypothetical X Values')
# ax1.set_xlabel('X Value')
# ax1.set_ylabel('Value')
# ax1.legend()
# ax1.grid(True)

# # 绘制第二组数据
# ax2.plot(x_2, new_design_data_2, marker='o', label='New Design Algorithm', linestyle='-')
# ax2.plot(x_2, old_design_data_2, marker='x', label='Old Design Algorithm', linestyle='-')
# ax2.set_title('Comparison of Data Usage by New and Old Design Algorithms')
# ax2.set_xlabel('Data Point Index')
# ax2.set_ylabel('Data Usage (GB)')
# ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
# ax2.legend()
# ax2.grid(True)

# # 显示图表
# plt.tight_layout()  # 调整子图之间的间距
# plt.show()


# import matplotlib.pyplot as plt

# # 第一个数据集
# new_design_times1 = [
#     378922, 557546, 583099, 819211, 846582, 1083640, 1109600, 1344340, 1371820, 
#     1606750, 1632060, 1866720, 1893730, 2132810, 2154060, 2392320, 2412110, 2645430
# ]
# x1 = [11000, 20000, 21000, 30000, 31000, 40000, 41000, 50000, 51000, 
#       60000, 61000, 70000, 71000, 80000, 81000, 90000, 91000, 100000]

# # 第二个数据集
# new_design_times2 = [
#     1.88915e+06, 215844, 286422, 369515, 463654, 568888, 686597, 814602, 814343,
#     943481, 900053, 818579, 922726, 1.0255e+06, 855119, 947738, 1.04458e+06,
#     1.141e+06, 1.24248e+06, 1.35052e+06, 1.46525e+06, 1.58304e+06, 1.74952e+06,
#     2.38073e+06, 3.58172e+06, 4.55835e+06
# ]
# x2 = [
#     5000, 6000, 7000, 8000, 9000, 10000,
#     11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000,
#     20000, 21000, 22000, 23000, 24000,
#     25000, 26000, 27000, 28000, 29000, 30000
# ]

# # 创建一个包含两个子图的图形
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# # 绘制第一个数据集
# ax1.plot(x1, new_design_times1, marker='o', linestyle='-', color='b', label='New Design Algorithm')
# ax1.set_title('Time Taken by New Design Algorithm (Set 1)')
# ax1.set_xlabel('Test/Iteration Number')
# ax1.set_ylabel('Time (assumed units)')
# ax1.legend()
# ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

# # 绘制第二个数据集，并设置y轴限制
# ax2.plot(x2, new_design_times2, marker='o', linestyle='-', color='b', label='New Design Algorithm')
# ax2.set_title('Time Taken by New Design Algorithm (Set 2)')
# ax2.set_xlabel('Test/Iteration Number')
# ax2.set_ylabel('Time (assumed units)')
# ax2.legend()
# ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
# ax2.set_ylim(bottom=0, top=max(new_design_times2) * 1.1)  # 设置y轴上限

# # 显示图形
# plt.tight_layout()  # 调整子图之间的间距
# plt.show()


















# m = 2000
# col = 100


# import matplotlib.pyplot as plt
# import numpy as np

# new_design_values_1 = [3.01785e+07, 5.65973e+07, 8.41229e+07, 1.08117e+08, 1.13602e+08, 1.30785e+08, 1.47589e+08, 1.98282e+08, 2.20129e+08, 2.43879e+08]
# old_design_values_1 = [6.25471e+07, 1.36683e+08, 2.39119e+08, 3.42238e+08, 4.68842e+08, 6.52117e+08, 8.9387e+08, 1.12683e+09, 1.41468e+09, 1.76528e+09]
# x_1 = np.arange(100, 1100, 100)

# new_design_data_2 = [
#     336330, 1.22686e+06, 2.77131e+06, 5.2022e+06, 8.99619e+06, 1.14117e+07, 1.63425e+07, 
#     1.99866e+07, 2.61709e+07, 3.2601e+07, 4.17899e+07, 4.9307e+07, 5.62417e+07, 
#     7.16404e+07, 7.39658e+07, 8.74493e+07, 9.68632e+07, 1.08865e+08, 1.31373e+08, 
#     1.3262e+08, 1.43879e+08, 1.72808e+08, 1.91541e+08, 1.90894e+08, 2.01113e+08
# ]
# old_design_data_2 = [
#     613866, 2.57858e+06, 5.67244e+06, 1.00037e+07, 1.57224e+07, 2.28275e+07, 3.07486e+07,
#     4.15605e+07, 5.0949e+07, 6.75636e+07, 7.61441e+07, 9.70095e+07, 1.13652e+08, 
#     1.41285e+08, 1.48954e+08, 1.65699e+08, 1.88489e+08, 2.07974e+08, 2.40617e+08,
#     2.58591e+08, 2.8509e+08, 3.24793e+08, 3.39759e+08, 3.82138e+08, 4.07191e+08
# ]
# x_2 = np.arange(200, 5200, 200)

# new_design_times1 = [
#     378922, 557546, 583099, 819211, 846582, 1083640, 1109600, 1344340, 1371820, 
#     1606750, 1632060, 1866720, 1893730, 2132810, 2154060, 2392320, 2412110, 2645430
# ]
# x3 = [11000, 20000, 21000, 30000, 31000, 40000, 41000, 50000, 51000, 
#       60000, 61000, 70000, 71000, 80000, 81000, 90000, 91000, 100000]
      
# new_design_times2 = [
#     1.88915e+06, 215844, 286422, 369515, 463654, 568888, 686597, 814602, 814343,
#     943481, 900053, 818579, 922726, 1.0255e+06, 855119, 947738, 1.04458e+06,
#     1.141e+06, 1.24248e+06, 1.35052e+06, 1.46525e+06, 1.58304e+06, 1.74952e+06,
#     2.38073e+06, 3.58172e+06, 4.55835e+06
# ]
# x4 = [
#     5000, 6000, 7000, 8000, 9000, 10000,
#     11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000,
#     20000, 21000, 22000, 23000, 24000,
#     25000, 26000, 27000, 28000, 29000, 30000
# ]

# fig, ((ax2, ax1), (ax4, ax3)) = plt.subplots(2, 2, figsize=(7, 7))

# # 绘制第一组数据到第一个子图
# ax1.plot(x_1, new_design_values_1, marker='o', label='GeLSA-CPU')
# ax1.plot(x_1, old_design_values_1, marker='x', label='eLSA')
# ax1.set_title('time series(m=2000)')
# ax1.set_xlabel('number of points')
# ax1.set_ylabel('time')
# ax1.legend()
# ax1.grid(True)

# # 绘制第二组数据到第二个子图
# ax2.plot(x_2, new_design_data_2, marker='o', label='GeLSA-CPU')
# ax2.plot(x_2, old_design_data_2, marker='x', label='eLSA')
# ax2.set_title('time series(n=100)')
# ax2.set_xlabel('number of item')
# ax2.legend()
# ax2.grid(True)

# # 绘制第三组数据到第三个子图
# ax3.plot(x3, new_design_times1, marker='o', label='GeLSA-GPU')
# ax3.set_title('time series(m=2000)')
# ax3.set_xlabel('number of points')
# ax3.set_ylabel('time')
# ax3.legend()
# ax3.grid(True)

# # 绘制第四组数据到第四个子图
# ax4.plot(x4, new_design_times2, marker='o', label='GeLSA-GPU')
# ax4.set_title('time series(n=100)')
# ax4.set_xlabel('number of item')

# ax4.legend()
# ax4.grid(True)

# plt.tight_layout()
# plt.show()

