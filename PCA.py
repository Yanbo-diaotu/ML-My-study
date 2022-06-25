from ast import If
from re import T
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def loadDataSet():
    # 数据存放位置：path-to-anaconda3\Lib\site-packages\sklearn\datasets\data
    iris = load_iris()
    # sepal length：萼片长度
    # sepal width：萼片宽度
    # petal length：花瓣长度
    # petal width：花瓣宽度
    print(iris.feature_names)
    # 0: 山鸢尾（setosa）
    # 1: 变色鸢尾（versicolor）
    # 2: 维吉尼亚鸢尾（virginica）
    print(iris.target_names)
    return iris.data, iris.target, iris.target_names

# 数据标准化
def Data_Standardization(x):
    std = StandardScaler()
    x_std = std.fit_transform(x)
    return x_std

# 确定目标维数
def Target_dimension(sorted_eig_vals):
    total_eig_vars = np.sum(sorted_eig_vals)
    cum_eig_vals = 0
    target_dim = -1
    threshold = 0.95
    for i in range(len(sorted_idx_eig_vals)):
        cum_eig_vals += sorted_eig_vals[i]
        print(cum_eig_vals)
        if(cum_eig_vals/total_eig_vars > threshold):
            target_dim = i+1
            print('目标维数为：', target_dim)
            break
    return target_dim

# 数据可视化
def Data_Visualization(x, y, y_names, Y, dimension):
    # 之前
    plt.figure(figsize=(6, 6))
    for lab, col in zip(('setosa', 'versicolor', 'virginica'), ('blue', 'red', 'green')):
        plt.scatter(x[y == list(y_names).index(lab), 0],
                    x[y == list(y_names).index(lab), 1],
                    label=lab,
                    c=col)
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    # 之后
    if dimension == 1:  # 1维使用此
        plt.figure(figsize=(6, 6))
        for lab, col in zip(('setosa', 'versicolor', 'virginica'), ('blue', 'red', 'green')):
            plt.scatter(Y[y == list(y_names).index(lab), 0],
                        Y[y == list(y_names).index(lab), 0],
                        label=lab,
                        c=col)
        plt.xlabel('New Feature 1')
        plt.ylabel('New Feature 1')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    elif dimension == 2:
        plt.figure(figsize=(6, 6))
        for lab, col in zip(('setosa', 'versicolor', 'virginica'), ('blue', 'red', 'green')):
            plt.scatter(Y[y == list(y_names).index(lab), 0],
                        Y[y == list(y_names).index(lab), 1],
                        label=lab,
                        c=col)
        plt.xlabel('New Feature 1')
        plt.ylabel('New Feature 2')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    elif dimension == 3:
        plt.figure(figsize=(6, 6))
        for lab, col in zip(('setosa', 'versicolor', 'virginica'), ('blue', 'red', 'green')):
            plt.scatter(Y[y == list(y_names).index(lab), 0],
                        Y[y == list(y_names).index(lab), 1],
                        label=lab,
                        c=col)
        plt.xlabel('New Feature 1')
        plt.ylabel('New Feature 2')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        for lab, col in zip(('setosa', 'versicolor', 'virginica'), ('blue', 'red', 'green')):
            plt.scatter(Y[y == list(y_names).index(lab), 0],
                        Y[y == list(y_names).index(lab), 2],
                        label=lab,
                        c=col)
        plt.xlabel('New Feature 1')
        plt.ylabel('New Feature 3')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        for lab, col in zip(('setosa', 'versicolor', 'virginica'), ('blue', 'red', 'green')):
            plt.scatter(Y[y == list(y_names).index(lab), 1],
                        Y[y == list(y_names).index(lab), 2],
                        label=lab,
                        c=col)
        plt.xlabel('New Feature 2')
        plt.ylabel('New Feature 3')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    x, y, y_names = loadDataSet()
    x_std = Data_Standardization(x)

    # 求协方差矩阵
    cov_mat = np.cov(x_std.T)
    print('Covariance matrix \n%s' % cov_mat)

    # 特征值分解
    # 第 i 列的特征向量 eig_vecs[:,i]对应第 i 个特征值 eig_vals[i]
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print('Eigenvectors \n%s' % eig_vecs)
    print('\nEigenvalues \n%s' % eig_vals)

    # 特征值排序，并得到相应的特征向量
        # 对特征值进行从大到小排序，返回排序后的索引
    sorted_idx_eig_vals = eig_vals.argsort() # 对特征值进行从小到大排序，返回排序后的索引
    sorted_idx_eig_vals = sorted_idx_eig_vals[-1::-1] # 索引取倒序
    print(sorted_idx_eig_vals)
        # 从大到小排序后的特征值
    sorted_eig_vals = eig_vals[sorted_idx_eig_vals]
        # 从大到小排序后的特征值对应的特征向量
    sorted_eig_vecs = eig_vecs[:, sorted_idx_eig_vals]

    # target_dim = Target_dimension(sorted_eig_vals)
    target_dim = 3
    
    # 构建特征向量矩阵𝑃
    # 因为前两维的特征值比较大，说明相应的数据变化比较大，所以我们决定降到 2 维数据，
    # 150x4->150x2 需要一个 4x2 的矩阵，前两维的特征向量组成映射矩阵
    P = sorted_eig_vecs[:, 0:target_dim]

    # 计算目标维度的特征
    Y = np.dot(P.T, x_std.T)
    Y = Y.T

    Data_Visualization(x, y, y_names, Y, target_dim)
