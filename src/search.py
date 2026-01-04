# 加载查看：
import numpy as np
theta = np.load('theta.npy')
print(theta.shape)  # 应该输出 (345,)
print(theta[:5])    # 示例: [12.5, 8.3, 15.7, 10.2, 9.8]