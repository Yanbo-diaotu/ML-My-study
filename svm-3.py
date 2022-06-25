import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import time

start = time.time()

data2 = pd.read_csv('data/svmdata2.csv')
data2.head()

positive = data2[data2['y'].isin([1])]
negative = data2[data2['y'].isin([0])]

# fig, ax = plt.subplots(figsize=(12, 8))
plt.figure(figsize=(12, 16))
plt.subplot(211)
plt.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
plt.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
plt.legend()

C = 10**2 # 2  4.6：score：1
kernel = 'poly' # 'rbf', 'poly',
# svc = svm.SVC(C=C, kernel=kernel, gamma=29, probability=True) # 默认核函数高斯核/RBF核 10
svc = svm.SVC(C=C, kernel=kernel, degree=6, probability=True) # 'poly' degree=3 time=16.5788

svc.fit(data2[['X1', 'X2']], data2['y'])

data2['Probability'] = svc.predict_proba(data2[['X1', 'X2']])[:, 0]

costtime = time.time() - start
print("Time:", costtime)

plt.subplot(212)
plt.scatter(data2['X1'], data2['X2'], s=30, c=data2['Probability'], cmap='winter')
plt.title("C = %e, kernel = %s, score= %.3f" %(C, kernel, svc.score(data2[['X1', 'X2']], data2['y'])))
plt.show()