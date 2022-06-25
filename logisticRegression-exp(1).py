# 根据两次考试成绩，利用逻辑回归模型预测学生是否被大学录取。
from copy import error
from numpy import *
import pandas as pd
import matplotlib.pyplot as plt
import random

from pyparsing import alphas # 随机模块

def load_data(file_name):
    data = pd.read_csv(file_name, header=None, names = ['Exam 1', 'Exam 2', 'Admitted'])
    print(data.head())
    data.insert(0, 'Ones', 1)
    print(data.head())

    X = data[['Ones', 'Exam 1', 'Exam 2']] # 训练样本
    y = data["Admitted"]                   # 标签

    return matrix(X), matrix(y).T

def computeCost(X, y, w):
    first = multiply(-y, X * w)            # 按位相乘
    second = log(1 + exp(X * w))

    return sum(first + second) / (len(X))  # len: 返回矩阵X的行数

def sigmoid(z):
    return 1 / (1 + exp(-z))

# 矩阵形式的梯度下降
def gradientDescentinMatrix(X, y, eta=0.001, iters=5000):
    num_samples, num_features = X.shape
    w = zeros((num_features, 1))

    for i in range(iters):
        error = sigmoid(X * w) - y
        term = X.T * error
        w = w - eta / num_samples * term   # update step

        cost = computeCost(X, y, w)
        #print('iters = %d, cost = %f' %(i, cost))

    return w

# 批量梯度下降
def gradientDescent(X, y, eta=0.001, iters=5000):
    num_samples, num_features = X.shape
    w = zeros((num_features, 1))

    for i in range(iters):
        temp = zeros((num_features, 1))
        error = sigmoid(X * w) - y
        for j in range(num_samples):
            temp = temp + error[j, 0] * X[j, :].T
        
        grad = temp / num_samples

        w = w - eta * grad                 # update step

        cost = computeCost(X, y, w)
        #print('iters = %d, cost = %f' %(i, cost))

    return w

# 随机梯度下降
def stochasticGradient1(X,  y, eta=0.001, iters=5000):
    num_samples, num_features = X.shape
    w = zeros((num_features, 1))

    for i in range(iters):
        for j in range(num_samples):
            grad = float(-y[j] + sigmoid(X[j, :] * w)) * X[j, :].T / num_samples
            w = w - eta * grad            # update step

        cost = computeCost(X, y, w)
        #print('iters = %d, cost = %f' %(i, cost))

    return w

#改进的随机梯度下降
def stochasticGradient2(X,  y, eta=0.001, iters=5000):
    num_samples, num_features = X.shape
    w = zeros((num_features, 1))

    for i in range(iters):
        for j in range(num_samples):
            alpha = 4 / (j + i + 1) + eta
            randIndex = int(random.uniform(0, num_samples))
            gard = float(-y[randIndex] + sigmoid(X[randIndex, :] * w)) * X[randIndex, :].T /num_samples
            w = w - alpha * gard
        
        cost = computeCost(X, y, w)
        #print('iters = %d, cost = %f' %(i, cost))

    return w

# plot the dots and predicted line
def plotBestFit(samples, labels, w):
    n = samples.shape[0]                   # 样本的个数
    xcord1=[]; ycord1=[]                   # list
    xcord2=[]; ycord2=[]                   # list
    for i in range(n):
        if int(labels[i])==1:
            xcord1.append(samples[i, 1])
            ycord1.append(samples[i, 2])
        else:
            xcord2.append(samples[i, 1])
            ycord2.append(samples[i, 2])

    fig=plt.figure()
    ax= fig.add_subplot(111)
    ax.scatter(xcord1, ycord1,s=30,c='red',marker='s')        # 绘制散点图 标记点是方形
    ax.scatter(xcord2, ycord2,s=30,c='green',marker='o')      # 绘制散点图 标记点是圆圈。marker默认是圆圈，也可以不指定'o'。
    x = arange(30, 100,  1)                                   # 对应x1轴的坐标序列
    w = array(w)
    y = (-w[0]-w[1] * x)/w[2]  # 公式来源：分类直线(x2即y) w0 + w1*x1 + w2*x2 = 0 变换后即 y = x2 = (-w[0]-w[1] * x)/w[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

def predict(w, X):
    probability = sigmoid(X * w)
    print('probability: ', probability)

    if probability >= 0.5:
        cls = 1
    else:
        cls = 0

    return cls

if __name__=='__main__':
    X, y = load_data('ex2data1.txt')

    eta = 0.001
    iters = 200000

    # w = gradientDescentinMatrix(X, y, eta, iters) # 矩阵形式的梯度下降
    # print('1Final weights: ', w.T)
    # plotBestFit(X, y, w)
    #
    # w = gradientDescent(X, y, eta, iters) # 批量梯度下降法
    # print('2Final weights: ', w.T)
    # plotBestFit(X, y, w)
    #
    # w = stochasticGradient1(X, y, eta, iters) # 随机梯度下降法
    # print('3Final weights: ', w.T)
    # plotBestFit(X, y, w)
    #
    w = stochasticGradient2(X, y, eta, iters) # 改进的随机梯度下降法
    print('4Final weights: ', w.T)
    plotBestFit(X, y, w)

    #w = matrix([-25.16131872,   0.20623159,   0.20147149]).T
    #plotBestFit(X, y, w)

    newSample = matrix([1, 60, 60])
    cls = predict(w, newSample)
    print(cls)