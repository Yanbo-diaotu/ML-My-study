from gc import disable
from turtle import distance
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

# 加载手写体数字
def loadDataSet():
    mnist = load_digits()
    x, test_x, y, test_y = train_test_split(mnist.data, mnist.target, test_size=0.25, random_state=40)
    print(mnist.keys())
    print(mnist.images.shape)
    plt.gray()
    return x, test_x, y, test_y

# 计算欧式距离
def calcEuclideanDistance(testdata, traindata):
    traindatasize = traindata.shape[0]                       # 计算训练集的长度
    # 将测试集扩展至训练集的长度，再求差值
    dif = np.tile(testdata, (traindatasize, 1)) - traindata
    sqrdif = dif ** 2                                        # 求差值的平方
    sumsqrdif = np.sum(sqrdif, axis=1)                       # 求平方和
    distance = np.sqrt(sumsqrdif)                            # 再开根号，即所有的距离
    return distance

# 计算曼哈顿距离
def calcManhattanDistance(testdata, traindata):
    traindatasize = traindata.shape[0]                       # 计算训练集的长度
    # 将测试集扩展至训练集的长度，再求差值
    dif = np.tile(testdata, (traindatasize, 1)) - traindata
    absdif = np.abs(dif)                                     # 求差值的绝对值
    distance = np.sum(absdif, axis=1)                        # 求绝对值的和
    return distance

# 计算切比雪夫距离
def calcChebyshevDistance(testdata, traindata):
    traindatasize = traindata.shape[0]                       # 计算训练集的长度
    # 将测试集扩展至训练集的长度，再求差值
    dif = np.tile(testdata, (traindatasize, 1)) - traindata
    absdif = np.abs(dif)                                     # 求差值的绝对值
    distance = np.max(absdif, axis=1)                        # 求绝对值的最大值
    return distance

def knn(k, testdata, traindata, labels):
    distance = calcEuclideanDistance(testdata, traindata)  # 计算欧式距离
    # distance = calcManhattanDistance(testdata, traindata)  # 计算曼哈顿距离
    # distance = calcChebyshevDistance(testdata, traindata)  # 计算切比雪夫距离
    # 对距离进行从小到大排序，返回排序后的索引
    sorted_distance = distance.argsort()
    count = {}                                               # 准备一个空字典，存放投票结果
    for i in range(0, k):
        # 提取索引多对应的标签值作为字典的 key
        vote = labels[sorted_distance[i]]
        # 票数作为字典的 value。若字典的 key 中没有 vote，get 函数返回 0，否则返回 key 对应的 value
        count[vote] = count.get(vote, 0) + 1
        # 对最后的投票结果进行降序排列
    sorted_count = sorted(count.items(), key=lambda x:x[1], reverse=True)
    return sorted_count[0][0]                                # 返回得票最多的标签

if __name__ == "__main__":
    x, test_x, y, test_y = loadDataSet()
    results = []
    for i in range(0, len(test_x)):
        result = knn(100, test_x[i], x, y)
        print("%d: 真正数字: %d, 测试结果为: %d" %(i, test_y[i], result))
        results.append(result)
    predictions = (results == test_y)
    accuracy = np.sum(predictions)/len(results)
    print('accuracy: ', accuracy)