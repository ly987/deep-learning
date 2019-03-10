'''

最简单的一个神经网络结构:
    实现xor操作

'''

import numpy as np

l1_num = 4      # l1层神经元个数为4
iter_times = 10000  # 迭代次数
input_feature = 2

# 1. 定义sigmoid函数，包括函数和函数的导数【deriv = False:表示函数，True：表示函数的导数】
def sigmoid(x,deriv = False):
    if(deriv):
        return x * (1-x)
    return 1.0/(1.0+np.exp(-x))

# 2. 定义输入x
x = np.array([[0, 0],
              [1, 0],
              [0, 1],
              [1, 1]]
)
# print(x.shape[1])      # (4,2)：4个样本，2种特征

# 3. 定义输出y
y = np.array([[0],
              [1],
              [1],
              [0]]
)
# print(y.shape)      # (4,1)：4个输出值

# 4. 定义初始化参数w，范围在（-1,1）
w0 = 2 * np.random.random((input_feature, l1_num)) - 1      #  (a,b):a表示上一层的特征数，b表示这一层的神经元数
w1 = 2 * np.random.random((l1_num, 1)) - 1
# print(w0)

# 5. 迭代
for i in range(iter_times):
    # 正向
    l0 = x
    l1 = sigmoid(np.dot(l0,w0))
    l2 = sigmoid(np.dot(l1, w1))
    l2_errors = y - l2

    if (i%1000) ==0:
        print("error:", l2_errors)
        print("predict:", l2)

    # 反向
    l2_delta = l2_errors * sigmoid(l2, deriv=True)
    l1_errors = l2_delta.dot(w1.T)
    l1_delta = l1_errors * sigmoid(l1, deriv=True)

    # 更新w
    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)


