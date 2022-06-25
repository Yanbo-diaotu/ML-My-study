# sklearn 库中导入 svm 模块
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# 定义三个样本和标签
X = np.array([[3, 3], [4, 3], [1,1]])
y = [1, 1, -1]

# 定义分类器
model = svm.SVC(kernel ='linear')  # 参数 kernel 为线性核函数

# 训练分类器
model.fit(X, y)  # 调用分类器的 fit 函数建立模型（即计算出划分超平面，且所有相关属性都保存在了分类器 model 里）

# 打印分类器 model的一系列参数
print(model)

# 支持向量
print(model.support_vectors_)

# 权重w
print(model.coef_[0])

# 偏置b
print(model.intercept_[0])

# 属于支持向量的点的index
print(model.support_)

# 在每一个类中有多少个点属于支持向量
print(model.n_support_)

# 获得划分超平面
# 划分超平面原方程：w0x0 + w1x1 + b = 0
# 将其转化为点斜式方程，并把x0看作 x，x1看作y，b看作w2
# 点斜式：y = -(w0/w1)x - (w2/w1)
w = model.coef_[0]  # w 是一个二维数据，coef 就是 w = [w0,w1]
k = -w[0] / w[1]  # 斜率
xx = np.linspace(-5, 5)  # 从 -5 到 5 产生一些连续的值（随机的）
# .intercept[0] 获得 bias，即 b 的值，b / w[1] 是截距
yy = k * xx - (model.intercept_[0]) / w[1]  # 带入 x 的值，获得直线方程

# 画出和划分超平面平行且经过支持向量的两条线（斜率相同，截距不同）
sv = model.support_vectors_[0] # 取出第一个支持向量点
yy_down = k * xx + (sv[1] - k * sv[0])
sv = model.support_vectors_[-1] # 取出最后一个支持向量点
yy_up = k * xx + (sv[1] - k * sv[0])

# 查看相关的参数值
print("w: ", w)
print("k: ", k)
print("support_vectors_: ", model.support_vectors_)
print("model.coef_: ", model.coef_)

# 在 scikit-learin 中，coef_ 保存了线性模型中划分超平面的参数向量。形式为(n_classes, n_features)。若 n_classes > 1，则为多分类问题，(1，n_features) 为二分类问题。

# 绘制划分超平面，边际平面和样本点
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# 圈出支持向量
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=180, facecolors='none', edgecolors='r') # facecolors='none':设置空心点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

plt.show()

# 预测新的样本
newSamples = [[2,0],[3,5]]

res = model.predict([[2,0],[3,5]])

for i in range(0,len(newSamples)):
    print('样本', newSamples[i], '的类别是：', res[i])