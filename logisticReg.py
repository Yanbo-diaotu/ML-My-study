import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costReg(w, X, y, learningRate):
    w = np.matrix(w)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * w.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * w.T)))
    reg = (learningRate /
           (2 * len(X))) * np.sum(np.power(w[:, 1:w.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

def gradientReg(w, X, y, learningRate):
    w = np.matrix(w)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(w.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * w.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + (
                (learningRate / len(X)) * w[:, i])

    return grad

def predict(w, X):
    probability = sigmoid(X * w.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def hfunc2(w, x1, x2):
    temp = w[0][0]
    place = 0
    for i in range(1, degree+1):
        for j in range(0, i+1):
            temp+= np.power(x1, i-j) * np.power(x2, j) * w[0][place+1]
            place+=1

    return temp

def find_decision_boundary(w):
    t1 = np.linspace(-1, 1.5, 1000)
    t2 = np.linspace(-1, 1.5, 1000)

    cordinates = [(x, y) for x in t1 for y in t2]

    # zip(*):解包
    x_cord, y_cord = zip(*cordinates)
    h_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    h_val['hval'] = hfunc2(w, h_val['x1'], h_val['x2'])

    decision = h_val[np.abs(h_val['hval']) < 2 * 10**-3]
    return decision.x1, decision.x2


path = 'ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
data.head()


positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test 1'],
           positive['Test 2'],
           s=50,
           c='b',
           marker='o',
           label='Accepted')
ax.scatter(negative['Test 1'],
           negative['Test 2'],
           s=50,
           c='r',
           marker='x',
           label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()


degree = 5

x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'Ones', 1)

# data.head()

for i in range(1, degree+1):
    for j in range(0, i+1):
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

data.head()

# drop(labels=None, axis=0, index=None, columns=None,
#      level=None, inplace=False, errors='raise'):
# 参数：
# labels ： 要删除的索引或列标签。
# axis ： {0或'index'，1或'columns'}，默认为0
# inplace ： bool，默认为False
# 如果为True，则执行就地操作并返回None。
data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)

data.head()


# set X and y (remember from above that we moved the label to column 0)
cols = data.shape[1]
X2 = data.iloc[:,1:cols]
y2 = data.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array w
X2 = np.array(X2.values)
y2 = np.array(y2.values)
w2 = np.zeros(cols-1)


regPara = 1
costReg(w2, X2, y2, regPara)
gradientReg(w2, X2, y2, regPara)


# scipy.optimize.fmin_tnc(func, x0, fprime=None, args=(), approx_grad=0, bounds=None, epsilon=1e-08, scale=None, offset=None, messages=15, maxCGit=-1, maxfun=None, eta=-1, stepmx=0, accuracy=0, fmin=0, ftol=-1, xtol=-1, pgtol=-1, rescale=-1, disp=None, callback=None)
# 参数：
# func：优化的目标函数
# x0：初值
# fprime：提供优化函数func的梯度函数，不然优化函数func必须返回函数值和梯度，或者设置approx_grad=True
# approx_grad :如果设置为True，会给出近似梯度
# args：元组，是传递给代价函数的参数
result2 = opt.fmin_tnc(func=costReg, x0=w2, fprime=gradientReg, args=(X2, y2, regPara))


w_min = np.matrix(result2[0])
predictions = predict(w_min, X2)
# correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]

# uppercase = ['A', 'B', 'C']
# lowercase = ['a', 'b', 'c']
# zipped = zip(uppercase, lowercase) # 打包
# [('A','a'), ('B','b'), ('C','c')]
# aa, bb = zip(*zipped) # 解包
# aa = ['A', 'B', 'C']
# bb = ['a', 'b', 'c']
correct = []
for (a, b) in zip(predictions, y2):
    if((a == 1 and b == 1) or (a == 0 and b == 0)):
        correct.append(1)
    else:
        correct.append(0)
print(correct)
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))


fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'],
           positive['Test 2'],
           s=50,
           c='b',
           marker='o',
           label='Accepted')
ax.scatter(negative['Test 1'],
           negative['Test 2'],
           s=50,
           c='r',
           marker='x',
           label='Rejected')
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')

x, y = find_decision_boundary(result2)
plt.scatter(x, y, c='y', s=10, label='Prediction')
ax.legend()
plt.show()