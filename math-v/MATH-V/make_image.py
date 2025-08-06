import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合保存图片不展示
import matplotlib.pyplot as plt
import numpy as np

# 假设有三组数据，每组10个点
arr1 = np.array([0.205,0.205,0.205,0.205,0.205])
arr2 = np.array([0.198,0.193,0.183,0.167,0.134])
arr3 = np.array([0.189,0.185,0.179,0.165,0.099])

x = np.array([0.3,0.4,0.5,0.6,0.75])

# 创建折线图
plt.plot(x, arr1, label='original', marker='o')
plt.plot(x, arr2, label='my', marker='s')
plt.plot(x, arr3, label='simple', marker='^')

# 添加标题和标签
plt.title('accuracy on math benchmark')
plt.xlabel('sparsity')
plt.ylabel('accuracy')

# 显示图例
plt.legend()

# 显示网格（可选）
plt.grid(True)

plt.savefig('./image.png')
