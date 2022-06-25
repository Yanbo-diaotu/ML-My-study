# 导入相关的包
import numpy as np
import pandas as pd
import pylab as plt  # 绘图功能
from sklearn import svm
import time

start = time.time()

data = pd.read_csv('data/softdata.txt', header=None, names=['X1', 'X2', 'y'])
print(data.head())

X = data[["X1", "X2"]].values # 训练样本
y = data["y"].values # 标签

X_outliers = np.array([[0, 2.3], [-1, 2.5], [-2, 2.5], [0, 0]])
y_outliers = np.array([0, 0, 1, 1])

X = np.concatenate((X, X_outliers), axis=0)
y = np.concatenate((y, y_outliers), axis=0)

# 建立 svm 模型
C = float("inf") # 2, 7.5, float("inf") best：7.5 
svc = svm.SVC(C=C, kernel='linear')
svc.fit(X, y)

# 获得划分超平面
# 划分超平面原方程：w0x0 + w1x1 + b = 0
# 将其转化为点斜式方程，并把 x0 看作 x，x1 看作 y，b 看作 w2
# 点斜式：y = -(w0/w1)x - (w2/w1)
w = svc.coef_[0]  # w 是一个二维数据，coef 就是 w = [w0,w1]
k = -w[0] / w[1]  # 斜率
xx = np.linspace(-5, 5)  # 从 -5 到 5 产生一些连续的值（随机的）
# .intercept[0] 获得 bias，即 b 的值，b / w[1] 是截距
yy = k * xx - (svc.intercept_[0]) / w[1]  # 带入 x 的值，获得直线方程

# 画出和划分超平面平行且经过支持向量的两条线（斜率相同，截距不同）
margin = 1/w[1]
yy_down = yy - margin
yy_up = yy + margin

# 查看相关的参数值
print("w: ", w)
print("k: ", k)
print("support_vectors_: ", svc.support_vectors_)
print("n_support_: ", svc.n_support_)
print("clf.coef_: ", svc.coef_)

# 在 scikit-learin 中，coef_ 保存了线性模型中划分超平面的参数向量。形式为(n_classes, n_features)。若 n_classes > 1，则为多分类问题，(1，n_features) 为二分类问题。

costtime = time.time() - start
print("Time:", costtime)

# 绘制划分超平面，边际平面和样本点
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
# 圈出支持向量
plt.scatter(svc.support_vectors_[0:svc.n_support_[0], 0], svc.support_vectors_[0:svc.n_support_[0], 1],
            s=180, facecolors='none', edgecolors='b')
plt.scatter(svc.support_vectors_[svc.n_support_[0]:, 0], svc.support_vectors_[svc.n_support_[0]:, 1],
            s=180, facecolors='none', edgecolors='r')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.title("C = %e, score= %.3f" %(C, svc.score(X, y)))

plt.axis('tight')
plt.show()