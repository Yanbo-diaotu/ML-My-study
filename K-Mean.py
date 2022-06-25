import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.cluster import KMeans


# 初始化质心
# X：输入特征；
# k：聚类簇数

def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    return centroids

# 为所有样本找到最近的质心
# X：样本；
# centroids：质心


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sqrt(np.sum((X[i, :] - centroids[j, :])**2))  # 欧式距离
            # dist = np.sum(X[i, :] - centroids[j, :])                # 曼哈顿距离
            # dist = np.max(np.abs(X[i, :] - centroids[j, :]))        # 切比雪夫距离
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx

# 重新计算质心
# X：样本；
# idx：对应样本的簇索引


def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        # np.where(idx == i)返回 idx == i 的索引
        # 返回值为元组，用[0]取元组中的第一维数据，并转为数组
        indices = np.where(idx == i)[0]
        centroids[i, :] = np.sum(X[indices, :], axis=0) / len(indices)
    return centroids

# k 均值算法核心函数
# X：样本；
# initial_centroids：初始质心；
# max_iters：最大迭代次数


def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
    return idx, centroids


if __name__ == "__main__":
    data = pd.read_csv('data.csv')
    plt.scatter(data['X1'], data['X2'])
    plt.show()
    X = data.values
    # 初始化 k 值
    K = 3
    # 随机初始化质心
    initial_centroids = init_centroids(X, K)

    # 运行 k 均值算法对 X 进行聚类，返回每个样本的簇索引和质心
    idx, centroids = run_k_means(X, initial_centroids, 10)

    # 可视化
    cluster1 = X[np.where(idx == 0)[0], :]
    cluster2 = X[np.where(idx == 1)[0], :]
    cluster3 = X[np.where(idx == 2)[0], :]
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.scatter(cluster1[:, 0], cluster1[:, 1],
               s=30, color='r', label='Cluster 1')
    ax.scatter(cluster2[:, 0], cluster2[:, 1],
               s=30, color='g', label='Cluster 2')
    ax.scatter(cluster3[:, 0], cluster3[:, 1],
               s=30, color='b', label='Cluster 3')
    ax.legend()
    plt.show()

    # '利用 SSE 选择 k'
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1, 9):
        estimator = KMeans(n_clusters=k)  # 构造聚类器
        estimator.fit(data)
        SSE.append(estimator.inertia_)  # estimator.inertia_：平方误差
    X = range(1, 9)
    plt.figure(figsize=(15, 10))
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X, SSE, 'o-')
    plt.show()

    # 读取图片
    # 图片大小为：(128, 128, 3)，图片有 128×128 个像素，每个像素有 RGB 3 个通道
    # 即有 128×128 个样本，每个样本有 3 个特征
    pic = img.imread('bird.png')
    plt.figure()
    plt.imshow(pic)
    plt.axis('off')
    plt.show()
    # 将样本维数调整为：(128×128, 3)，即有 128×128 个样本，每个样本有 3 个特征
    X = np.reshape(pic, (pic.shape[0] * pic.shape[1], pic.shape[2]))
    # 初始化 k 值
    k = 4
    # 随机初始化质心
    initial_centroids = init_centroids(X, k)

    # 运行 k 均值算法对 X 进行聚类，返回每个样本的簇索引和质心
    idx, centroids = run_k_means(X, initial_centroids, 3)

    # 为每个样本找到离最终质心最新的簇索引
    idx = find_closest_centroids(X, centroids)

    # 将质心的值赋给同簇的像素
    # a = np.array([[3, 3, 3], [6, 6, 6], [9, 9, 9]])
    # b = a[[0, 1, 1, 2, 2, 2], :]
    # print(b)
    # [[3 3 3]
    # [6 6 6]
    # [6 6 6]
    # [9 9 9]
    # [9 9 9]
    # [9 9 9]]
    X_recovered = centroids[idx.astype(int), :]
    # 将压缩图片恢复到原始大小
    X_recovered = np.reshape(
        X_recovered, (pic.shape[0], pic.shape[1], pic.shape[2]))
    # 可视化
    plt.imshow(X_recovered)
    plt.axis('off')
    plt.show()
