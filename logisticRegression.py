from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_name):
    data = pd.read_csv(file_name, header=None, names = ['Exam 1', 'Exam 2', 'Admitted'])
    print(data.head())
    data.insert(0, 'Ones', 1)
    print(data.head())

    X = data[['Ones', 'Exam 1', 'Exam 2']] # 训练样本
    y = data["Admitted"] # 标签

    return matrix(X), matrix(y).T

def computeCost(X, y, w):
    first = multiply(-y, X * w)
    second = log(1 + exp(X * w))

    return sum(first + second) / (len(X)) # len: 返回矩阵X的行数

def sigmoid(z):
    return 1 / (1 + exp(-z))

def gradient(X, y, w):
    num_samples, num_features = X.shape
    grad = zeros((num_features, 1))
    temp = zeros((num_features, 1))

    error = sigmoid(X * w) - y

    for i in range(num_samples):
        temp = temp + error[i, 0] * X[i, :].T

    grad = temp / num_samples

    return grad

# plot the dots and predicted line
def plotBestFit(samples, labels, w):
    # plotBestFit(samples, labels, weights)
    # samples = array([[1, 1, 1],
    #                                  [3, 4, 1],
    #                                  [3, 3, 1]])
    #
    # labels = array([[1], [1], [-1]])
    n = samples.shape[0] # 样本的个数
    xcord1=[]; ycord1=[]  # list
    xcord2=[]; ycord2=[]  # list
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
    ax.scatter(xcord2, ycord2,s=30,c='green',marker='o')   # 绘制散点图 标记点是圆圈。marker默认是圆圈，也可以不指定'o'。
    x = arange(30, 100,  1) # 对应x1轴的坐标序列
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

    num_samples, num_features = X.shape
    w = zeros((num_features, 1))
    cost = computeCost(X, y, w)
    print('initial cost: ', cost)

    eta = 0.001
    iters =  100000

    for i in range(iters):
        w = w - eta * gradient(X, y, w)  # update step

        # compute the gradient using matrix operation
        # error = sigmoid(X * w) - y
        # term = X.T * error
        # w = w - eta / num_samples * term # update step

        cost = computeCost(X, y, w)
        print('iters = %d, cost = %f' %(i, cost))

    print('Final weights: ', w.T)

    # w = matrix([-25.16131872,   0.20623159,   0.20147149]).T
    plotBestFit(X, y, w)

    newSample = matrix([1, 45, 45])
    cls = predict(w, newSample)
    print(cls)