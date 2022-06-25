# 根据城市人口数量，预测开小吃店的利润
from numpy import * # 科学计算的核心库
import pandas as pd # 数据分析库
import matplotlib.pyplot as plt # 可视化库

def load_data(file_name):
    data = pd.read_csv(file_name, header=None, names=['Population', 'Profit'])
    print(data.head())
    data.insert(0, 'Ones', 1)
    print(data.head())

    X = data[["Ones", "Population"]] # 训练样本
    y = data["Profit"] # 标签

    return matrix(X), matrix(y).T

# 单变量线性回归，分别计算w和b的闭式解 (close-form solution, 解析解)
def regression1(X, y):
    X_mat_avg = mean(X, axis = 0) # 压缩行，实值
    X_mat_tmp = X - X_mat_avg
    num_samples = X.shape[0]

    # multiply: 按位相乘
    w = sum(multiply(y, X_mat_tmp)) / (sum(power(X, 2)) - 1 / num_samples * power(sum(X), 2))
    b = 1/num_samples*sum(y - w * X)

    ws = append([b], [w], axis = 0) # 一维数组
    return matrix(ws).T # matrix(ws)：行向量

# 单变量线性回归，向量形式计算闭式解 (close-form solution, 解析解)
def regression2(X, y):
    xTx = X.T * X
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse!\n")
        return

    ws = xTx.I * (X.T * y)
    return ws

def computeCost(X, y, w):
    inner = power((y - (X * w) ), 2)

    return sum(inner) / (2 * X.shape[0])

# gradient descent (numerical solution，数值解)
def regression3(X, y):
    num_samples, num_features  = X.shape
    ws = zeros((num_features, 1))

    eta = 0.01 # 0.01
    iters = 5000
    costs = zeros(iters)

    for i in range(iters):
        error = y - (X * ws)
        term = X.T * error
        ws = ws + (eta *2 / num_samples * term) # update step

        cost = computeCost(X, y, ws)
        costs[i] = cost
        # print('iters = %d, cost = %f' %(i, cost))

    return ws, costs

# plot the dots and predicted line
def plotLine(X, y, ws):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(X[:, 1].flatten().A[0], y[:, 0].flatten().A[0])
    ax.scatter(array(X[:, 1]), array(y[:, 0]))
    # print(squeeze(array(X[:, 1])).shape)
    x_copy = X.copy()
    x_copy.sort(0)
    y_hat = x_copy * ws
    ax.plot(x_copy[:, 1], y_hat, color='red')
    plt.show()

# plot the curve w.r.t. costs
def plotCosts(costs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(arange(costs.shape[0]), costs, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    plt.show()

if __name__=='__main__':
    X_mat, y_mat = load_data('ex1data1.txt')

    ws = regression1(X_mat[:, 1], y_mat)
    print('w for regression 1: ', ws)
    plotLine(X_mat, y_mat, ws)

    ws = regression2(X_mat, y_mat)
    print('w for regression 2: ', ws)
    plotLine(X_mat, y_mat, ws)

    ws, costs = regression3(X_mat, y_mat)
    print('w for regression 3: ', ws)
    plotLine(X_mat, y_mat, ws)
    plotCosts(costs)