import numpy as np
import matplotlib.pyplot as plot

hidden_layer_size = 2  # 隐层单元个数
input_number = 2    # 输入单元个数
output_number = 1   # 输出值个数
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])      # 数据集
y = np.array([[0], [1], [1], [0]])  # 正确的输出值

# 初始化需要用到的权重，权值范围为（-1,1）【由于np.random.rand（）随机数范围是0~1，所以使用了epsilon常数】
def rand_init_weights(L_in, L_out, epsilon):
    epsilon_init = epsilon
    w = np.random.rand(L_out, 1+L_in) * 2 * epsilon_init - epsilon_init
    return w

# 激活函数sigmoid
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# 对sigmoid函数的倒数
def sigmoid_gradient(z):
    g = np.multiply(sigmoid(z), (1-sigmoid(z)))
    return g

# 损失函数
def nn_cost_function(theta1, theta2, X, y):
    m = X.shape[0]                  # 数据集中第一个数组中的个数 4
    D_1 = np.zeros(theta1.shape)    # [[0, 0, 0], [0, 0, 0]]
    D_2 = np.zeros(theta2.shape)    # [0, 0, 0]
    h_total = np.zeros((m, 1))      # 存储所有样本的预测值    [0, 0, 0, 0]
    # 计算所有参数的偏导数
    for t in range(m):
        a_1 = np.vstack((np.array([[1]]), X[t:t + 1, :].T))     # 构造输入  [[1], [0], [0]]
        z_2 = np.dot(theta1, a_1)   # 加权后第一层输出值    [[a1],[a2]]
        a_2 = np.vstack((np.array([[1]]), sigmoid(z_2)))    # 构造隐层的输入   [[1], [a1]. [a2]]
        z_3 = np.dot(theta2, a_2)   # 加权后的第二层输出值   [[a3]]
        a_3 = sigmoid(z_3)  # 对第二层的值sigmoid一下，得到输出值h
        h = a_3
        h_total[t,0] = h
        delta_3 = h - y[t:t + 1, :].T   # 最后一层与实际值的偏差值
        # theta2[:, 1:].T  截取第二个和第三个数
        delta_2 = np.multiply(np.dot(theta2[:, 1:].T, delta_3), sigmoid_gradient(z_2))  # 隐层与实际值的偏差值
        D_2 = D_2 + np.dot(delta_3, a_2.T)  # 隐层所有参数的误差
        D_1 = D_1 + np.dot(delta_2, a_1.T)  # 最后一层所有参数的误差
    theta1_grad = (1.0 / m) * D_1   # 第一层参数的偏导数，取所有样本中参数的均值，没有加正则项
    theta2_grad = (1.0 / m) * D_2
    J = (1.0 / m) * np.sum(-y * np.log(h_total) - (np.array([[1]]) - y) * np.log(1 - h_total))

    return {'theta1_grad': theta1_grad, 'theta2_grad': theta2_grad, 'J': J, 'h': h_total}

# 之前的问题之一，epsilon的值设置的太小

theta1 = rand_init_weights(input_number, hidden_layer_size, epsilon=1)
theta2 = rand_init_weights(hidden_layer_size, output_number, epsilon=1)

# 之前的问题之二，迭代次数太少
iter_times = 10000
# 之前的问题之三，学习率太小
alpha = 0.5
result = {'J': [], 'h': []}
theta_s = {}
for i in range(iter_times):
    cost_fun_result = nn_cost_function(theta1=theta1, theta2=theta2, X=x, y=y)
    theta1_g = cost_fun_result.get('theta1_grad')
    theta2_g = cost_fun_result.get('theta2_grad')
    J = cost_fun_result.get('J')
    h_current = cost_fun_result.get('h')
    theta1 -= alpha * theta1_g
    theta2 -= alpha * theta2_g
    result['J'].append(J)
    result['h'].append(h_current)
    if i == 0 or i == (iter_times - 1):
        print('theta1', theta1)
    print('theta2', theta2)
    theta_s['theta1_' + str(i)] = theta1.copy()
    theta_s['theta2_' + str(i)] = theta2.copy()

plot.plot(result.get('J'))
plot.show()
print(theta_s)
print(result.get('h')[0], result.get('h')[-1])
