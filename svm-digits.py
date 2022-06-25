import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import pickle as pickle

mnist = load_digits()
x,test_x,y,test_y = train_test_split(mnist.data,mnist.target,test_size=0.35,random_state=40) #0.25 40

print(mnist.keys())
print(mnist.images.shape)

# plt.gray() #

# show images
# for i in range(0, 10):
#    plt.imshow(mnist.images[i])
#    plt.show()

# model = svm.LinearSVC() #

# kernel:
# 'linear'：线性核函数
# 'poly'：多项式核函数
# 'rbf'：径像核函数/高斯核
# 'sigmoid'：sigmoid核函数
model = svm.SVC(kernel='poly')
model.fit(x, y)

print(model.decision_function_shape)
# dec = model.decision_function(x) #

plt.imshow(test_x[0].reshape(8, 8))
plt.show()
z = model.predict([test_x[0]])
print('识别结果:', z)

z = model.predict(test_x)
print('准确率:',np.sum(z==test_y)/z.size)

# 保存模型
with open('./model.pkl', 'wb') as file:
    pickle.dump(model, file)

# 加载模型
with open('./model.pkl','rb') as file:
    model = pickle.load(file)

plt.imshow(test_x[9].reshape(8, 8))
plt.show()
z = model.predict([test_x[9]])
print('识别结果:', z)