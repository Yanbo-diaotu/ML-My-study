# coding: Utf-8
# by Liu Yingjie and Liu Yufan CNU-IE
# version: 2022/7/8 final

# 代码索引
# 加载数据集
# 分割数据集
# 标准化数据集
# 选择KNN算法K值
# KNN算法
# 选择SVM算法C值 
# 选择SVM算法gamma值
# SVM算法
# 选择RF算法森林树的个数
# RF算法
# 高斯朴素贝叶斯算法
# 决策树算法
# 逻辑回归算法
# 比较决策树与随机森林
# 比较逻辑回归与支持向量机（ovr）
# 比较SVM（sigmoid核函数）与MLP（sigmoid激活函数）
# 算法参数选择比较

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# 加载数据集
def Load_dataset_digits():
    dataset = load_digits()
    x_initial = dataset.data
    y_initial = dataset.target
    return x_initial, y_initial

# 分割数据集
def Split_dataset(x_initial, y_initial):
    x, x_test, y, y_test = train_test_split(x_initial, y_initial, test_size=0.2, random_state=None)  # test_size: 0.2 random_state: 32
    return x, x_test, y, y_test

# 标准化数据集
def Standard_dataset(x, x_test):
    # 预处理数据 标准化
    ss = preprocessing.StandardScaler()
    train_SS_x = ss.fit_transform(x)
    test_SS_x = ss.transform(x_test)
    return train_SS_x, test_SS_x

# 选择相对最优的K值
def Choose_K(x_train, x_test, y_train, y_test):
    k_range = range(2, 17)
    k_wrong_test = []
    k_wrong_train = []
    min_test = []
    K_temp = []
    min_wrong = 2
    min_dif = 2
    K_choose = 0
    # 遍历预设的K值
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(x_train, y_train)
        temp_test = 1 - knn.score(x_test, y_test)
        temp_train = 1 - knn.score(x_train, y_train)
        k_wrong_test.append(temp_test)
        k_wrong_train.append(temp_train)
    # 找错小
        if temp_test <= min_wrong:
            min_wrong = temp_test
            min_test.append(temp_test)
            K_temp.append(k)
    # 找差小
    for i in range(len(min_test)):
        dif = abs(min_test[i] - min_wrong)
        if dif <= min_dif:
            min_dif = dif
            K_choose = K_temp[i]

    plt.plot(k_range, k_wrong_test, ls="-", lw=2 , label="Test")
    plt.plot(k_range, k_wrong_train, ls="--", lw=2 , label="Train")
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Wrong')
    plt.legend()
    plt.show()
    print("The best K is: ", K_choose)
    return K_choose

def Choose_K_(x_train, x_test, y_train, y_test):
    k_range = range(2, 17)
    k_wrong_test = []
    k_wrong_train = []
    min_test = []
    K_temp = []
    min_wrong = 2
    min_dif = 2
    K_choose = 0
    # 遍历预设的K值
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(x_train, y_train)
        temp_test = 1 - knn.score(x_test, y_test)
        temp_train = 1 - knn.score(x_train, y_train)
        k_wrong_test.append(temp_test)
        k_wrong_train.append(temp_train)
    # 找错小
        if temp_test <= min_wrong:
            min_wrong = temp_test
            min_test.append(temp_test)
            K_temp.append(k)
    # 找差小
    for i in range(len(min_test)):
        dif = abs(min_test[i] - min_wrong)
        if dif <= min_dif:
            min_dif = dif
            K_choose = K_temp[i]
    
    return K_choose

def KNN(x_train, x_test, y_train, y_test, k):
    # 创建KNN分类器
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="kd_tree", p=2, n_jobs=-1)
    # K个邻居 邻居的标签权重 近的邻居加权   构建方式 kd树    距离度量 p=2/1/inf  并行数
    # weights = "distance"/"uniform" algorithm="auto"/"ball_tree"/"kd_tree"/"brute" p=2/1/np.inf n_jobs=-1/None
    knn.fit(x_train, y_train)                   # 训练
    y_predict = knn.predict(x_test)             # 预测
    Accuracy_rate = knn.score(x_test, y_test)   # 准确率
    # print("Accuracy(KNN): ", Accuracy_rate)
    return y_predict, Accuracy_rate

def Choose_C(x_train, x_test, y_train, y_test):
    c_range = range(-2, 4)
    c_wrong_test = []
    c_wrong_train = []
    min_test = []
    c_temp = []
    min_wrong = 2
    min_dif = 2
    c_choose = 0
    # 遍历预设的C值
    for i in c_range:
        c = 10 ** i
        svm = SVC(C=c, kernel="poly", degree=3, gamma="auto")
        svm.fit(x_train, y_train)
        temp_test = 1 - svm.score(x_test, y_test)
        temp_train = 1 - svm.score(x_train, y_train)
        c_wrong_test.append(temp_test)
        c_wrong_train.append(temp_train)
        # 找错小
        if temp_test <= min_wrong:
            min_wrong = temp_test
            min_test.append(temp_test)
            c_temp.append(i)
        # 找差小
    for i in range(len(min_test)):
        dif = abs(min_test[i] - min_wrong)
        if dif <= min_dif:
            min_dif = dif
            c_choose = c_temp[i]


    plt.plot(c_range, c_wrong_test, ls="-", lw=2 , label="Test")
    plt.plot(c_range, c_wrong_train, ls="--", lw=2 , label="Train")
    plt.xlabel('Value of C for SVM(C=10^x)') 
    plt.ylabel('Wrong')
    plt.legend()
    plt.show()
    print("The best C is: ", c_choose)
    return c_choose

def Choose_C_gamma(x_train, x_test, y_train, y_test):
    c_range = range(-2, 4)
    c_wrong_test = []
    c_wrong_train = []
    g_range = range(-4, 2)
    g_wrong_test = []
    g_wrong_train = []
    min_test = []
    min_test_g = []
    c_temp = []
    g_temp = []
    min_wrong = 2
    min_dif = 2
    min_wrong_g = 2
    min_dif_g = 2
    c_choose = 0
    g_choose = 0
    # 遍历预设的C值
    for i in c_range:
        c = 10 ** i
        svm_c = SVC(C=c, kernel="poly", degree=3, gamma="auto")
        svm_c.fit(x_train, y_train)
        temp_test_c = 1 - svm_c.score(x_test, y_test)
        temp_train_c = 1 - svm_c.score(x_train, y_train)
        c_wrong_test.append(temp_test_c)
        c_wrong_train.append(temp_train_c)
        # 找错小
        if temp_test_c <= min_wrong:
            min_wrong = temp_test_c
            min_test.append(temp_test_c)
            c_temp.append(i)
    for j in g_range:
        g = 10 ** j
        svm_g = SVC(C=1, kernel="poly", degree=3, gamma=g)
        svm_g.fit(x_train, y_train)
        temp_test_g = 1 - svm_g.score(x_test, y_test)
        temp_train_g = 1 - svm_g.score(x_train, y_train)
        g_wrong_test.append(temp_test_g)
        g_wrong_train.append(temp_train_g)
        if temp_test_g <= min_wrong_g:
            min_wrong_g = temp_test_g
            min_test_g.append(temp_test_g)
            g_temp.append(j)
    # 找差小
    for i in range(len(min_test)):
        dif = abs(min_test[i] - min_wrong)
        if dif <= min_dif:
            min_dif = dif
            c_choose = c_temp[i]
    for j in range(len(min_test_g)):
        dif = abs(min_test_g[j] - min_wrong_g)
        if dif <= min_dif_g:
            min_dif_g = dif
            g_choose = g_temp[j]

    return c_choose, g_choose

def Choose_gamma(x_train, x_test, y_train, y_test):
    g_range = range(-4, 2)
    g_wrong_test = []
    g_wrong_train = []
    min_test = []
    g_temp = []
    min_wrong = 2
    min_dif = 2
    g_choose = 0
    # 遍历预设的gamma值
    for i in g_range:
        g = 10 ** i
        svm = SVC(C=1, kernel="poly", degree=3, gamma=g)
        svm.fit(x_train, y_train)
        temp_test = 1 - svm.score(x_test, y_test)
        temp_train = 1 - svm.score(x_train, y_train)
        g_wrong_test.append(temp_test)
        g_wrong_train.append(temp_train)
        # 找错小
        if temp_test < min_wrong:
            min_wrong = temp_test
            min_test.append(temp_test)
            g_temp.append(g)
        # 找差小
    for i in range(len(min_test)):
        dif = abs(min_test[i] - min_wrong)
        if dif < min_dif:
            min_dif = dif
            g_choose = g_temp[i]

    plt.plot(g_range, g_wrong_test, ls="-", lw=2 , label="Test")
    plt.plot(g_range, g_wrong_train, ls="--", lw=2 , label="Train")
    plt.xlabel('Value of gamma for SVM(gamma=10^x)')
    plt.ylabel('Wrong')
    plt.legend()
    plt.show()
    print("The best gamma is: ", g_choose)
    return g_choose

def SVM(x_train, x_test, y_train, y_test, c, g):
    # 创建SVM分类器
    svm = SVC(C=c, kernel='poly', degree=3, gamma=g, coef0=0.0, shrinking=True, decision_function_shape='ovr')
    # 惩罚系数   核函数poly   多项式次数   函数gamma   常数系数0.0   收缩的启发式方法   分类算法 一对多ovr
    # kernel = 'linear'/'poly'/'rbf'/'sigmoid' degree=3 gamma='auto' C=1.0/0.1/0.01/0.001/0.0001
    # coef0=0.0/0.1/0.5/1/2 shrinking=True/False decision_function_shape='ovo'/'ovr'
    svm.fit(x_train, y_train)                  # 训练
    y_predict = svm.predict(x_test)            # 预测
    Accuracy_rate = svm.score(x_test, y_test)  # 准确率
    # print("Accuracy(SVM): ", Accuracy_rate)
    return y_predict, Accuracy_rate


def Choose_N(x_train, x_test, y_train, y_test):
    n_range = range(1, 17)
    n_wrong_test = []
    n_wrong_train = []
    min_test = []
    time_temp = []
    min_v = 2
    n_choose = 0
    # 遍历预设的n值
    for i in n_range:
        n = i * 32
        strat_time = time.time()
        rf = RandomForestClassifier(n_estimators=n)
        rf.fit(x_train, y_train)
        costtime = time.time() - strat_time
        time_temp.append(costtime)
        temp_test = 1 - rf.score(x_test, y_test)
        temp_train = 1 - rf.score(x_train, y_train)
        n_wrong_test.append(temp_test)
        n_wrong_train.append(temp_train)                  # 训练集错误率0.00%
        min_test = temp_test * 0.5 + costtime * 0.1 * 0.5 # 程序运行时间乘0.1进行压缩，保证与测试集错误率在同一比较级下
        # print(temp_test)
        # print(costtime * 0.1)
        # print(min_test)
        # print("**********")
        if min_test <= min_v:
            min_v = min_test
            n_choose = n

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(n_range, n_wrong_test, ls="-", lw=2 , label="Test")
    plt.plot(n_range, n_wrong_train, ls="--", lw=2 , label="Train")
    plt.xlabel('Value of n_estimators for RF') 
    plt.ylabel('Wrong')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(n_range, time_temp, ls="-", lw=2 , label="Time")
    plt.xlabel('Value of n_estimators for RF')
    plt.ylabel('Time')
    plt.legend()
    plt.show()
    print("The best n_estimators is: ", n_choose)
    return n_choose

def Choose_N_(x_train, x_test, y_train, y_test):
    n_range = range(1, 17)
    n_wrong_test = []
    n_wrong_train = []
    min_test = []
    time_temp = []
    min_v = 2
    n_choose = 0
    # 遍历预设的n值
    for i in n_range:
        n = i * 32
        strat_time = time.time()
        rf = RandomForestClassifier(n_estimators=n)
        rf.fit(x_train, y_train)
        costtime = time.time() - strat_time
        time_temp.append(costtime)
        temp_test = 1 - rf.score(x_test, y_test)
        temp_train = 1 - rf.score(x_train, y_train)
        n_wrong_test.append(temp_test)
        n_wrong_train.append(temp_train)                  # 训练集错误率0.00%
        min_test = temp_test * 0.5 + costtime * 0.1 * 0.5 # 程序运行时间乘0.1进行压缩，保证与测试集错误率在同一比较级下
        # print(temp_test)
        # print(costtime * 0.1)
        # print(min_test)
        # print("**********")
        if min_test <= min_v:
            min_v = min_test
            n_choose = n

    return n_choose

def Random_forest(x_train, x_test, y_train, y_test, n):
    # 创建随机森林分类器
    rf = RandomForestClassifier(n_estimators=n, criterion="gini", max_depth=5, max_features="auto", bootstrap=True, n_jobs=-1)
    # 森林中200棵决策树   决策树的构建方式 gini/entropy   决策树深度5   决策树的最大特征数量auto   有放回地随机选取样本   并行数
    # n_estimators: 500/200/100/50/10/1 criterion="gini"/"entropy" max_depth=5/3/1/'None'
    # max_features="auto"/"log2" bootstrap=True/False n_jobs=-1/None
    rf.fit(x_train, y_train)                  # 训练
    y_predict = rf.predict(x_test)            # 预测
    Accuracy_rate = rf.score(x_test, y_test)  # 准确率
    # print("Accuracy(Random forest): ", Accuracy_rate)
    return y_predict, Accuracy_rate


# 高斯朴素贝叶斯分类，不调整参数，作为平行对比
def Naive_bayes(x_train, x_test, y_train, y_test):
    # 创建朴素贝叶斯分类器
    nb = GaussianNB()
    nb.fit(x_train, y_train)                 # 训练
    y_predict = nb.predict(x_test)           # 预测
    Accuracy_rate = nb.score(x_test, y_test) # 准确率
    # print("Accuracy(Naive bayes): ", Accuracy_rate)
    return y_predict, Accuracy_rate

# 决策树分类，作为随机森林的对比
def DecisionTree(x_train, x_test, y_train, y_test):
    # 创建决策树分类器
    dt = DecisionTreeClassifier(criterion="gini", max_depth=5, max_features="auto")
    # 结点划分度量标准 基尼系数   决策树深度5   决策树的最大特征数量(允许搜索的最大属性个数) auto(sqrt(n_features))
    # criterion="gini"/"entropy"  max_depth=5/4/3/2/1/None  max_features="auto"/"log2"
    dt.fit(x_train, y_train)                 # 训练
    y_predict = dt.predict(x_test)           # 预测
    Accuracy_rate = dt.score(x_test, y_test) # 准确率
    # print("Accuracy(Decision tree): ", Accuracy_rate)
    return y_predict, Accuracy_rate

# 逻辑回归分类，作为SVM的对比，对比ovr的效果
def Logistic_Regression(x_train, x_test, y_train, y_test):
    # 创建LogisticRegression分类器
    lr = LogisticRegression(penalty="l2", C=0.1, solver="liblinear", multi_class="ovr")
    # 惩罚项L2    正则化系数λ的倒数    优化算法选择参数     多分类方式ovr
    # c=1/0.1/0.01/0.001/0.0001    solver="liblinear"/"lbfgs"/"newton-cg"/"sag"/"saga"
    lr.fit(x_train, y_train)                 # 训练
    y_predict = lr.predict(x_test)           # 预测
    Accuracy_rate = lr.score(x_test, y_test) # 准确率
    # print("Accuracy(LR): ", Accuracy_rate)
    return y_predict, Accuracy_rate

# 多层感知机，对比SVM核函数选择sigmoid的效果
def MLP(x_train, x_test, y_train, y_test):
    # 创建MLP分类器
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver='lbfgs')
    # 隐藏层个数及神经元个数    激活函数 logistic(sigmoid)    优化函数 lbfgs 小数据集收敛更快
    # activation="identity"/"logistic"/"tanh"/"relu" 默认relu  solver="lbfgs"/"sgd"/"adam"默认adam
    mlp.fit(x_train, y_train)                 # 训练
    y_predict = mlp.predict(x_test)           # 预测
    Accuracy_rate = mlp.score(x_test, y_test) # 准确率
    # print("Accuracy(MLP): ", Accuracy_rate)
    return y_predict, Accuracy_rate


# 决策树与随机森林对比(随机森林中有200棵)
def Decision_tree_Random_forest(train_SS_x, test_SS_x, y_train, y_test):
    time_RF = []
    time_DT = []
    Accuracy_rate_RF = []
    Accuracy_rate_DT = []
    for i in range(0, 100):
        start_rf = time.time()
        y_predict_rf, Accuracy_rate_rf = Random_forest(train_SS_x, test_SS_x, y_train, y_test, 200)
        costtime_rf = time.time() - start_rf
        time_RF.append(costtime_rf)
        Accuracy_rate_RF.append(Accuracy_rate_rf)

        start_dt = time.time()
        y_predict_dt, Accuracy_rate_dt = DecisionTree(train_SS_x, test_SS_x, y_train, y_test)
        costtime_dt = time.time() - start_dt
        time_DT.append(costtime_dt)
        Accuracy_rate_DT.append(Accuracy_rate_dt)
    
    # 求平均运行时间及准确率
    Ave_time_RF = np.mean(time_RF, dtype=np.longdouble)
    Ave_time_DT = np.mean(time_DT, dtype=np.longdouble)
    Ave_Accuracy_rate_RF = np.mean(Accuracy_rate_RF, dtype=np.longdouble)
    Ave_Accuracy_rate_DT = np.mean(Accuracy_rate_DT, dtype=np.longdouble)
    print("Average time of Random Forest: ", Ave_time_RF)
    print("Average time of Decision Tree: ", Ave_time_DT)
    print("Average Accuracy rate of Random Forest: ", Ave_Accuracy_rate_RF)
    print("Average Accuracy rate of Decision Tree: ", Ave_Accuracy_rate_DT)
    
    # 可视化数据
    n_range = range(0, 100)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(n_range, Accuracy_rate_RF, ls="-", lw=2 , label="RF")
    plt.plot(n_range, Accuracy_rate_DT, ls="--", lw=2 , label="DT")
    plt.xlabel('The x-th operation') 
    plt.ylabel('Accuracy rate')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(n_range, time_RF, ls="-", lw=2 , label="RF")
    plt.plot(n_range, time_DT, ls="--", lw=2 , label="DT")
    plt.xlabel('The x-th operation')
    plt.ylabel('Time')
    plt.legend()
    plt.suptitle("Random Forest and Decision Tree")
    plt.show()
    return

def Logistic_Regression_SVM(train_SS_x, test_SS_x, y_train, y_test):
    time_LR = []
    time_SVM = []
    Accuracy_rate_LR = []
    Accuracy_rate_SVM = []
    for i in range(0, 100):
        start_LR = time.time()
        y_predict_lr, Accuracy_rate_lr = Logistic_Regression(train_SS_x, test_SS_x, y_train, y_test)
        costtime_LR = time.time() - start_LR
        time_LR.append(costtime_LR)
        Accuracy_rate_LR.append(Accuracy_rate_lr)

        start_svm = time.time()
        y_predict_svm, Accuracy_rate_svm = SVM(train_SS_x, test_SS_x, y_train, y_test, 1, "auto")
        costtime_svm = time.time() - start_svm
        time_SVM.append(costtime_svm)
        Accuracy_rate_SVM.append(Accuracy_rate_svm)

    # 求平均运行时间及准确率
    Ave_time_RF = np.mean(time_LR, dtype=np.longdouble)
    Ave_time_DT = np.mean(time_SVM, dtype=np.longdouble)
    Ave_Accuracy_rate_RF = np.mean(Accuracy_rate_LR, dtype=np.longdouble)
    Ave_Accuracy_rate_DT = np.mean(Accuracy_rate_SVM, dtype=np.longdouble)
    print("Average time of Logistic Regression: ", Ave_time_RF)
    print("Average time of SVM: ", Ave_time_DT)
    print("Average Accuracy rate of Logistic Regression: ", Ave_Accuracy_rate_RF)
    print("Average Accuracy rate of SVM: ", Ave_Accuracy_rate_DT)
    
    # 可视化数据
    n_range = range(0, 100)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(n_range, Accuracy_rate_LR, ls="-", lw=2 , label="LR")
    plt.plot(n_range, Accuracy_rate_SVM, ls="--", lw=2 , label="SVM")
    plt.xlabel('The x-th operation') 
    plt.ylabel('Accuracy rate')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(n_range, time_LR, ls="-", lw=2 , label="LR")
    plt.plot(n_range, time_SVM, ls="--", lw=2 , label="SVM")
    plt.xlabel('The x-th operation')
    plt.ylabel('Time')
    plt.legend()
    plt.suptitle("Logistic Regression and SVM")
    plt.show()
    return

def MLP_SVM(train_SS_x, test_SS_x, y_train, y_test):
    time_MLP = []
    time_SVM = []
    Accuracy_rate_MLP = []
    Accuracy_rate_SVM = []
    for i in range(0, 100):
        start_MLP = time.time()
        y_predict_mlp, Accuracy_rate_mlp = MLP(train_SS_x, test_SS_x, y_train, y_test)
        costtime_MLP = time.time() - start_MLP
        time_MLP.append(costtime_MLP)
        Accuracy_rate_MLP.append(Accuracy_rate_mlp)

        start_svm = time.time()
        svm = SVC(kernel='sigmoid')
        svm.fit(train_SS_x, y_train)
        y_predict_svm = svm.predict(test_SS_x)
        Accuracy_rate_svm = svm.score(test_SS_x, y_test)
        costtime_svm = time.time() - start_svm
        time_SVM.append(costtime_svm)
        Accuracy_rate_SVM.append(Accuracy_rate_svm)

    # 求平均运行时间及准确率
    Ave_time_MLP = np.mean(time_MLP, dtype=np.longdouble)
    Ave_time_SVM = np.mean(time_SVM, dtype=np.longdouble)
    Ave_Accuracy_rate_MLP = np.mean(Accuracy_rate_MLP, dtype=np.longdouble)
    Ave_Accuracy_rate_SVM = np.mean(Accuracy_rate_SVM, dtype=np.longdouble)
    print("Average time of MLP: ", Ave_time_MLP)
    print("Average time of SVM: ", Ave_time_SVM)
    print("Average Accuracy rate of MLP: ", Ave_Accuracy_rate_MLP)
    print("Average Accuracy rate of SVM: ", Ave_Accuracy_rate_SVM)

    # 可视化数据
    n_range = range(0, 100)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(n_range, Accuracy_rate_MLP, ls="-", lw=2 , label="MLP")
    plt.plot(n_range, Accuracy_rate_SVM, ls="--", lw=2 , label="SVM")
    plt.xlabel('The x-th operation')
    plt.ylabel('Accuracy rate')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(n_range, time_MLP, ls="-", lw=2 , label="MLP")
    plt.plot(n_range, time_SVM, ls="--", lw=2 , label="SVM")
    plt.xlabel('The x-th operation')
    plt.ylabel('Time')
    plt.legend()
    plt.suptitle("MLP and SVM")
    plt.show()
    return

def Algorithm_parameter_comparison(train_SS_x, test_SS_x, y_train, y_test):
    print("1. KNN algorithm")
    print("2. KNN weights")
    print("3. KNN Distance measurement method")
    print("4. SVM shrinking")
    print("5. SVM decision_function_shape")
    print("6. SVM kernel")
    print("7. RF criterion")
    print("8. RF max_depth")
    print("9. RF max_features")
    num = input("Please choose: ")
    x, y = Load_dataset_digits()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)
    train_SS_x, test_SS_x = Standard_dataset(x_train, x_test)
    
    # KNN algorithm
    if num == "1":
        time_auto = []
        time_ball_tree = []
        time_kd_tree = []
        time_brute = []
        accuracy_rate_auto = []
        accuracy_rate_ball_tree = []
        accuracy_rate_kd_tree = []
        accuracy_rate_brute = []
        for i in range(0, 100):
            start_auto = time.time()
            k_auto = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto", p=2)
            k_auto.fit(train_SS_x, y_train)
            y_predict_auto = k_auto.predict(test_SS_x)
            Accuracy_rate_auto = k_auto.score(test_SS_x, y_test)
            costtime_auto = time.time() - start_auto
            time_auto.append(costtime_auto)
            accuracy_rate_auto.append(Accuracy_rate_auto)

            start_ball_tree = time.time()
            k_ball_tree = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="ball_tree", p=2)
            k_ball_tree.fit(train_SS_x, y_train)
            y_predict_ball_tree = k_ball_tree.predict(test_SS_x)
            Accuracy_rate_ball_tree = k_ball_tree.score(test_SS_x, y_test)
            costtime_ball_tree = time.time() - start_ball_tree
            time_ball_tree.append(costtime_ball_tree)
            accuracy_rate_ball_tree.append(Accuracy_rate_ball_tree)

            start_kd_tree = time.time()
            k_kd_tree = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="kd_tree", p=2)
            k_kd_tree.fit(train_SS_x, y_train)
            y_predict_kd_tree = k_kd_tree.predict(test_SS_x)
            Accuracy_rate_kd_tree = k_kd_tree.score(test_SS_x, y_test)
            costtime_kd_tree = time.time() - start_kd_tree
            time_kd_tree.append(costtime_kd_tree)
            accuracy_rate_kd_tree.append(Accuracy_rate_kd_tree)

            start_brute = time.time()
            k_brute = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="brute", p=2)
            k_brute.fit(train_SS_x, y_train)
            y_predict_brute = k_brute.predict(test_SS_x)
            Accuracy_rate_brute = k_brute.score(test_SS_x, y_test)
            costtime_brute = time.time() - start_brute
            time_brute.append(costtime_brute)
            accuracy_rate_brute.append(Accuracy_rate_brute)

        Ave_time_auto = np.mean(time_auto, dtype=np.longdouble)
        Ave_time_ball_tree = np.mean(time_ball_tree, dtype=np.longdouble)
        Ave_time_kd_tree = np.mean(time_kd_tree, dtype=np.longdouble)
        Ave_time_brute = np.mean(time_brute, dtype=np.longdouble)
        Ave_Accuracy_rate_auto = np.mean(accuracy_rate_auto, dtype=np.longdouble)
        Ave_Accuracy_rate_ball_tree = np.mean(accuracy_rate_ball_tree, dtype=np.longdouble)
        Ave_Accuracy_rate_kd_tree = np.mean(accuracy_rate_kd_tree, dtype=np.longdouble)
        Ave_Accuracy_rate_brute = np.mean(accuracy_rate_brute, dtype=np.longdouble)
        print("Average time(auto): ", Ave_time_auto)
        print("Average time(ball_tree): ", Ave_time_ball_tree)
        print("Average time(kd_tree): ", Ave_time_kd_tree)
        print("Average time(brute): ", Ave_time_brute)
        print("Average Accuracy rate(auto): ", Ave_Accuracy_rate_auto)
        print("Average Accuracy rate(ball_tree): ", Ave_Accuracy_rate_ball_tree)
        print("Average Accuracy rate(kd_tree): ", Ave_Accuracy_rate_kd_tree)
        print("Average Accuracy rate(brute): ", Ave_Accuracy_rate_brute)

        plt.figure(figsize=(10, 8))
        num_range = range(0, 100)
        plt.subplot(2, 1, 1)
        plt.plot(num_range, time_auto, ls="-", lw=2, label="auto")
        plt.plot(num_range, time_ball_tree, ls="-.", lw=2, label="ball_tree")
        plt.plot(num_range, time_kd_tree, ls="--", lw=2, label="kd_tree")
        plt.plot(num_range, time_brute, ls=":", lw=2, label="brute")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Running time")
        plt.subplot(2, 1, 2)
        plt.plot(num_range, accuracy_rate_auto, ls="-", lw=2, label="auto")
        plt.plot(num_range, accuracy_rate_ball_tree, ls="-.", lw=2, label="ball_tree")
        plt.plot(num_range, accuracy_rate_kd_tree, ls="--", lw=2, label="kd_tree")
        plt.plot(num_range, accuracy_rate_brute, ls=":", lw=2, label="brute")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy rate")
        plt.suptitle("KNN algorithm")
        plt.show()
    
    # KNN weights
    elif num == "2":
        time_distance = []
        time_uniform = []
        accuracy_rate_distance = []
        accuracy_rate_uniform = []
        for i in range(0, 100):
            start_distance = time.time()
            k_distance = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="auto", p=2)
            k_distance.fit(train_SS_x, y_train)
            y_predict_distance = k_distance.predict(test_SS_x)
            Accuracy_rate_distance = k_distance.score(test_SS_x, y_test)
            costtime_distance = time.time() - start_distance
            time_distance.append(costtime_distance)
            accuracy_rate_distance.append(Accuracy_rate_distance)

            start_uniform = time.time()
            k_uniform = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto", p=2)
            k_uniform.fit(train_SS_x, y_train)
            y_predict_uniform = k_uniform.predict(test_SS_x)
            Accuracy_rate_uniform = k_uniform.score(test_SS_x, y_test)
            costtime_uniform = time.time() - start_uniform
            time_uniform.append(costtime_uniform)
            accuracy_rate_uniform.append(Accuracy_rate_uniform)
        
        Ave_time_distance = np.mean(time_distance, dtype=np.longdouble)
        Ave_time_uniform = np.mean(time_uniform, dtype=np.longdouble)
        Ave_Accuracy_rate_distance = np.mean(accuracy_rate_distance, dtype=np.longdouble)
        Ave_Accuracy_rate_uniform = np.mean(accuracy_rate_uniform, dtype=np.longdouble)
        print("Average time(distance): ", Ave_time_distance)
        print("Average time(uniform): ", Ave_time_uniform)
        print("Average Accuracy rate(distance): ", Ave_Accuracy_rate_distance)
        print("Average Accuracy rate(uniform): ", Ave_Accuracy_rate_uniform)

        plt.figure(figsize=(10, 8))
        num_range = range(0, 100)
        plt.subplot(2, 1, 1)
        plt.plot(num_range, time_distance, ls="-", lw=2, label="distance")
        plt.plot(num_range, time_uniform, ls="-.", lw=2, label="uniform")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Running time")
        plt.subplot(2, 1, 2)
        plt.plot(num_range, accuracy_rate_distance, ls="-", lw=2, label="distance")
        plt.plot(num_range, accuracy_rate_uniform, ls="-.", lw=2, label="uniform")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy rate")
        plt.suptitle("KNN weights")
        plt.show()
    
    # KNN p
    elif num == "3":
        time_euclidean = []
        time_manhattan = []
        time_chebyshev = []
        accuracy_rate_euclidean = []
        accuracy_rate_manhattan = []
        accuracy_rate_chebyshev = []
        for i in range(0, 100):
            start_euclidean = time.time()
            k_euclidean = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="auto", p=2)
            k_euclidean.fit(train_SS_x, y_train)
            y_predict_euclidean = k_euclidean.predict(test_SS_x)
            Accuracy_rate_euclidean = k_euclidean.score(test_SS_x, y_test)
            costtime_euclidean = time.time() - start_euclidean
            time_euclidean.append(costtime_euclidean)
            accuracy_rate_euclidean.append(Accuracy_rate_euclidean)

            start_manhattan = time.time()
            k_manhattan = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="auto", p=1)
            k_manhattan.fit(train_SS_x, y_train)
            y_predict_manhattan = k_manhattan.predict(test_SS_x)
            Accuracy_rate_manhattan = k_manhattan.score(test_SS_x, y_test)
            costtime_manhattan = time.time() - start_manhattan
            time_manhattan.append(costtime_manhattan)
            accuracy_rate_manhattan.append(Accuracy_rate_manhattan)

            start_chebyshev = time.time()
            k_chebyshev = KNeighborsClassifier(n_neighbors=5, weights="distance", algorithm="auto", p=np.inf)
            k_chebyshev.fit(train_SS_x, y_train)
            y_predict_chebyshev = k_chebyshev.predict(test_SS_x)
            Accuracy_rate_chebyshev = k_chebyshev.score(test_SS_x, y_test)
            costtime_chebyshev = time.time() - start_chebyshev
            time_chebyshev.append(costtime_chebyshev)
            accuracy_rate_chebyshev.append(Accuracy_rate_chebyshev)
        
        Ave_time_euclidean = np.mean(time_euclidean, dtype=np.longdouble)
        Ave_time_manhattan = np.mean(time_manhattan, dtype=np.longdouble)
        Ave_time_chebyshev = np.mean(time_chebyshev, dtype=np.longdouble)
        Ave_Accuracy_rate_euclidean = np.mean(accuracy_rate_euclidean, dtype=np.longdouble)
        Ave_Accuracy_rate_manhattan = np.mean(accuracy_rate_manhattan, dtype=np.longdouble)
        Ave_Accuracy_rate_chebyshev = np.mean(accuracy_rate_chebyshev, dtype=np.longdouble)
        print("Average time(euclidean): ", Ave_time_euclidean)
        print("Average time(manhattan): ", Ave_time_manhattan)
        print("Average time(chebyshev): ", Ave_time_chebyshev)
        print("Average Accuracy rate(euclidean): ", Ave_Accuracy_rate_euclidean)
        print("Average Accuracy rate(manhattan): ", Ave_Accuracy_rate_manhattan)
        print("Average Accuracy rate(chebyshev): ", Ave_Accuracy_rate_chebyshev)

        plt.figure(figsize=(10, 8))
        num_range = range(0, 100)
        plt.subplot(2, 1, 1)
        plt.plot(num_range, time_euclidean, ls="-", lw=2, label="euclidean")
        plt.plot(num_range, time_manhattan, ls="-.", lw=2, label="manhattan")
        plt.plot(num_range, time_chebyshev, ls="--", lw=2, label="chebyshev")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Running time")
        plt.subplot(2, 1, 2)
        plt.plot(num_range, accuracy_rate_euclidean, ls="-", lw=2, label="euclidean")
        plt.plot(num_range, accuracy_rate_manhattan, ls="-.", lw=2, label="manhattan")
        plt.plot(num_range, accuracy_rate_chebyshev, ls="--", lw=2, label="chebyshev")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy rate")
        plt.suptitle("KNN Distance measurement method")
        plt.show()
    
    # SVM shrinking
    elif num == "4":
        time_True = []
        time_False = []
        accuracy_rate_True = []
        accuracy_rate_False = []
        for i in range(0, 100):
            start_True = time.time()
            svm_True = SVC(kernel="rbf", C=1, shrinking=True)
            svm_True.fit(train_SS_x, y_train)
            y_predict_True = svm_True.predict(test_SS_x)
            Accuracy_rate_True = svm_True.score(test_SS_x, y_test)
            costtime_True = time.time() - start_True
            time_True.append(costtime_True)
            accuracy_rate_True.append(Accuracy_rate_True)

            start_False = time.time()
            svm_False = SVC(kernel="rbf", C=1, shrinking=False)
            svm_False.fit(train_SS_x, y_train)
            y_predict_False = svm_False.predict(test_SS_x)
            Accuracy_rate_False = svm_False.score(test_SS_x, y_test)
            costtime_False = time.time() - start_False
            time_False.append(costtime_False)
            accuracy_rate_False.append(Accuracy_rate_False)
        
        Ave_time_True = np.mean(time_True, dtype=np.longdouble)
        Ave_time_False = np.mean(time_False, dtype=np.longdouble)
        Ave_Accuracy_rate_True = np.mean(accuracy_rate_True, dtype=np.longdouble)
        Ave_Accuracy_rate_False = np.mean(accuracy_rate_False, dtype=np.longdouble)
        print("Average time(True): ", Ave_time_True)
        print("Average time(False): ", Ave_time_False)
        print("Average Accuracy rate(True): ", Ave_Accuracy_rate_True)
        print("Average Accuracy rate(False): ", Ave_Accuracy_rate_False)

        plt.figure(figsize=(10, 8))
        num_range = range(0, 100)
        plt.subplot(2, 1, 1)
        plt.plot(num_range, time_True, ls="-", lw=2, label="True")
        plt.plot(num_range, time_False, ls="-.", lw=2, label="False")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Running time")
        plt.subplot(2, 1, 2)
        plt.plot(num_range, accuracy_rate_True, ls="-", lw=2, label="True")
        plt.plot(num_range, accuracy_rate_False, ls="-.", lw=2, label="False")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy rate")
        plt.suptitle("SVM shrinking")
        plt.show()
    
    # SVM decision_function_shape
    elif num == "5":
        time_ovr = []
        time_ovo = []
        accuracy_rate_ovr = []
        accuracy_rate_ovo = []
        for i in range(0, 100):
            start_ovr = time.time()
            svm_ovr = SVC(kernel="rbf", C=1, decision_function_shape="ovr")
            svm_ovr.fit(train_SS_x, y_train)
            y_predict_ovr = svm_ovr.predict(test_SS_x)
            Accuracy_rate_ovr = svm_ovr.score(test_SS_x, y_test)
            costtime_ovr = time.time() - start_ovr
            time_ovr.append(costtime_ovr)
            accuracy_rate_ovr.append(Accuracy_rate_ovr)

            start_ovo = time.time()
            svm_ovo = SVC(kernel="rbf", C=1, decision_function_shape="ovo")
            svm_ovo.fit(train_SS_x, y_train)
            y_predict_ovo = svm_ovo.predict(test_SS_x)
            Accuracy_rate_ovo = svm_ovo.score(test_SS_x, y_test)
            costtime_ovo = time.time() - start_ovo
            time_ovo.append(costtime_ovo)
            accuracy_rate_ovo.append(Accuracy_rate_ovo)

        Ave_time_ovr = np.mean(time_ovr, dtype=np.longdouble)
        Ave_time_ovo = np.mean(time_ovo, dtype=np.longdouble)
        Ave_Accuracy_rate_ovr = np.mean(accuracy_rate_ovr, dtype=np.longdouble)
        Ave_Accuracy_rate_ovo = np.mean(accuracy_rate_ovo, dtype=np.longdouble)
        print("Average time(ovr): ", Ave_time_ovr)
        print("Average time(ovo): ", Ave_time_ovo)
        print("Average Accuracy rate(ovr): ", Ave_Accuracy_rate_ovr)
        print("Average Accuracy rate(ovo): ", Ave_Accuracy_rate_ovo)

        plt.figure(figsize=(10, 8))
        num_range = range(0, 100)
        plt.subplot(2, 1, 1)
        plt.plot(num_range, time_ovr, ls="-", lw=2, label="ovr")
        plt.plot(num_range, time_ovo, ls="-.", lw=2, label="ovo")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Running time")
        plt.subplot(2, 1, 2)
        plt.plot(num_range, accuracy_rate_ovr, ls="-", lw=2, label="ovr")
        plt.plot(num_range, accuracy_rate_ovo, ls="-.", lw=2, label="ovo")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy rate")
        plt.suptitle("SVM decision_function_shape")
        plt.show()

    # SVM kernel
    elif num == "6":
        time_rbf = []
        time_poly = []
        accuracy_rate_rbf = []
        accuracy_rate_poly = []
        for i in range(0, 100):
            start_rbf = time.time()
            svm_rbf = SVC(kernel="rbf")
            svm_rbf.fit(train_SS_x, y_train)
            y_predict_rbf = svm_rbf.predict(test_SS_x)
            Accuracy_rate_rbf = svm_rbf.score(test_SS_x, y_test)
            costtime_rbf = time.time() - start_rbf
            time_rbf.append(costtime_rbf)
            accuracy_rate_rbf.append(Accuracy_rate_rbf)

            start_poly = time.time()
            svm_poly = SVC(kernel="poly")
            svm_poly.fit(train_SS_x, y_train)
            y_predict_poly = svm_poly.predict(test_SS_x)
            Accuracy_rate_poly = svm_poly.score(test_SS_x, y_test)
            costtime_poly = time.time() - start_poly
            time_poly.append(costtime_poly)
            accuracy_rate_poly.append(Accuracy_rate_poly)
        
        Ave_time_rbf = np.mean(time_rbf, dtype=np.longdouble)
        Ave_time_poly = np.mean(time_poly, dtype=np.longdouble)
        Ave_Accuracy_rate_rbf = np.mean(accuracy_rate_rbf, dtype=np.longdouble)
        Ave_Accuracy_rate_poly = np.mean(accuracy_rate_poly, dtype=np.longdouble)
        print("Average time(rbf): ", Ave_time_rbf)
        print("Average time(poly): ", Ave_time_poly)
        print("Average Accuracy rate(rbf): ", Ave_Accuracy_rate_rbf)
        print("Average Accuracy rate(poly): ", Ave_Accuracy_rate_poly)

        plt.figure(figsize=(10, 8))
        num_range = range(0, 100)
        plt.subplot(2, 1, 1)
        plt.plot(num_range, time_rbf, ls="-", lw=2, label="rbf")
        plt.plot(num_range, time_poly, ls="-.", lw=2, label="poly")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Running time")
        plt.subplot(2, 1, 2)
        plt.plot(num_range, accuracy_rate_rbf, ls="-", lw=2, label="rbf")
        plt.plot(num_range, accuracy_rate_poly, ls="-.", lw=2, label="poly")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy rate")
        plt.suptitle("SVM kernel")
        plt.show()
    
    # RF criterion
    elif num == "7":
        time_gini = []
        time_entropy = []
        accuracy_rate_gini = []
        accuracy_rate_entropy = []
        for i in range(0, 100):
            start_gini = time.time()
            rf_gini = RandomForestClassifier(criterion="gini")
            rf_gini.fit(x_train, y_train)
            y_predict_gini = rf_gini.predict(x_test)
            Accuracy_rate_gini = rf_gini.score(x_test, y_test)
            costtime_gini = time.time() - start_gini
            time_gini.append(costtime_gini)
            accuracy_rate_gini.append(Accuracy_rate_gini)

            start_entropy = time.time()
            rf_entropy = RandomForestClassifier(criterion="entropy")
            rf_entropy.fit(x_train, y_train)
            y_predict_entropy = rf_entropy.predict(x_test)
            Accuracy_rate_entropy = rf_entropy.score(x_test, y_test)
            costtime_entropy = time.time() - start_entropy
            time_entropy.append(costtime_entropy)
            accuracy_rate_entropy.append(Accuracy_rate_entropy)
        
        Ave_time_gini = np.mean(time_gini, dtype=np.longdouble)
        Ave_time_entropy = np.mean(time_entropy, dtype=np.longdouble)
        Ave_Accuracy_rate_gini = np.mean(accuracy_rate_gini, dtype=np.longdouble)
        Ave_Accuracy_rate_entropy = np.mean(accuracy_rate_entropy, dtype=np.longdouble)
        print("Average time(gini): ", Ave_time_gini)
        print("Average time(entropy): ", Ave_time_entropy)
        print("Average Accuracy rate(gini): ", Ave_Accuracy_rate_gini)
        print("Average Accuracy rate(entropy): ", Ave_Accuracy_rate_entropy)

        plt.figure(figsize=(10, 8))
        num_range = range(0, 100)
        plt.subplot(2, 1, 1)
        plt.plot(num_range, time_gini, ls="-", lw=2, label="gini")
        plt.plot(num_range, time_entropy, ls="-.", lw=2, label="entropy")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Running time")
        plt.subplot(2, 1, 2)
        plt.plot(num_range, accuracy_rate_gini, ls="-", lw=2, label="gini")
        plt.plot(num_range, accuracy_rate_entropy, ls="-.", lw=2, label="entropy")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy rate")
        plt.suptitle("Random Forest criterion")
        plt.show()
    
    # RF max_depth
    elif num == "8":
        time_5 = []
        time_3 = []
        time_1 = []
        time_None = []
        accuracy_rate_5 = []
        accuracy_rate_3 = []
        accuracy_rate_1 = []
        accuracy_rate_None = []
        for i in range(0, 100):
            start_5 = time.time()
            rf_5 = RandomForestClassifier(max_depth=5)
            rf_5.fit(x_train, y_train)
            y_predict_5 = rf_5.predict(x_test)
            Accuracy_rate_5 = rf_5.score(x_test, y_test)
            costtime_5 = time.time() - start_5
            time_5.append(costtime_5)
            accuracy_rate_5.append(Accuracy_rate_5)

            start_3 = time.time()
            rf_3 = RandomForestClassifier(max_depth=3)
            rf_3.fit(x_train, y_train)
            y_predict_3 = rf_3.predict(x_test)
            Accuracy_rate_3 = rf_3.score(x_test, y_test)
            costtime_3 = time.time() - start_3
            time_3.append(costtime_3)
            accuracy_rate_3.append(Accuracy_rate_3)

            start_1 = time.time()
            rf_1 = RandomForestClassifier(max_depth=1)
            rf_1.fit(x_train, y_train)
            y_predict_1 = rf_1.predict(x_test)
            Accuracy_rate_1 = rf_1.score(x_test, y_test)
            costtime_1 = time.time() - start_1
            time_1.append(costtime_1)
            accuracy_rate_1.append(Accuracy_rate_1)

            start_None = time.time()
            rf_None = RandomForestClassifier(max_depth=None)
            rf_None.fit(x_train, y_train)
            y_predict_None = rf_None.predict(x_test)
            Accuracy_rate_None = rf_None.score(x_test, y_test)
            costtime_None = time.time() - start_None
            time_None.append(costtime_None)
            accuracy_rate_None.append(Accuracy_rate_None)
        
        Ave_time_5 = np.mean(time_5, dtype=np.longdouble)
        Ave_time_3 = np.mean(time_3, dtype=np.longdouble)
        Ave_time_1 = np.mean(time_1, dtype=np.longdouble)
        Ave_time_None = np.mean(time_None, dtype=np.longdouble)
        Ave_Accuracy_rate_5 = np.mean(accuracy_rate_5, dtype=np.longdouble)
        Ave_Accuracy_rate_3 = np.mean(accuracy_rate_3, dtype=np.longdouble)
        Ave_Accuracy_rate_1 = np.mean(accuracy_rate_1, dtype=np.longdouble)
        Ave_Accuracy_rate_None = np.mean(accuracy_rate_None, dtype=np.longdouble)
        print("Average time(5): ", Ave_time_5)
        print("Average time(3): ", Ave_time_3)
        print("Average time(1): ", Ave_time_1)
        print("Average time(None): ", Ave_time_None)
        print("Average Accuracy rate(5): ", Ave_Accuracy_rate_5)
        print("Average Accuracy rate(3): ", Ave_Accuracy_rate_3)
        print("Average Accuracy rate(1): ", Ave_Accuracy_rate_1)
        print("Average Accuracy rate(None): ", Ave_Accuracy_rate_None)

        plt.figure(figsize=(10, 8))
        num_range = range(0, 100)
        plt.subplot(2, 1, 1)
        plt.plot(num_range, time_5, ls="-", lw=2, label="max_depth=5")
        plt.plot(num_range, time_3, ls="-.", lw=2, label="max_depth=3")
        plt.plot(num_range, time_1, ls="--", lw=2, label="max_depth=1")
        plt.plot(num_range, time_None, ls=":", lw=2, label="max_depth=None")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Running time")
        plt.subplot(2, 1, 2)
        plt.plot(num_range, accuracy_rate_5, ls="-", lw=2, label="max_depth=5")
        plt.plot(num_range, accuracy_rate_3, ls="-.", lw=2, label="max_depth=3")
        plt.plot(num_range, accuracy_rate_1, ls="--", lw=2, label="max_depth=1")
        plt.plot(num_range, accuracy_rate_None, ls=":", lw=2, label="max_depth=None")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy rate")
        plt.suptitle("Random Forest max_depth")
        plt.show()
    
    # RF max_features
    elif num == "9":
        tmie_auto = []
        tmie_log2 = []
        tmie_None = []
        accuracy_rate_auto = []
        accuracy_rate_log2 = []
        accuracy_rate_None = []
        for i in range(0, 100):
            start_auto = time.time()
            rf_auto = RandomForestClassifier(max_depth=5, max_features="auto")
            rf_auto.fit(x_train, y_train)
            y_predict_auto = rf_auto.predict(x_test)
            Accuracy_rate_auto = rf_auto.score(x_test, y_test)
            costtime_auto = time.time() - start_auto
            tmie_auto.append(costtime_auto)
            accuracy_rate_auto.append(Accuracy_rate_auto)

            start_log2 = time.time()
            rf_log2 = RandomForestClassifier(max_depth=5, max_features="log2")
            rf_log2.fit(x_train, y_train)
            y_predict_log2 = rf_log2.predict(x_test)
            Accuracy_rate_log2 = rf_log2.score(x_test, y_test)
            costtime_log2 = time.time() - start_log2
            tmie_log2.append(costtime_log2)
            accuracy_rate_log2.append(Accuracy_rate_log2)

            start_None = time.time()
            rf_None = RandomForestClassifier(max_depth=5, max_features=None)
            rf_None.fit(x_train, y_train)
            y_predict_None = rf_None.predict(x_test)
            Accuracy_rate_None = rf_None.score(x_test, y_test)
            costtime_None = time.time() - start_None
            tmie_None.append(costtime_None)
            accuracy_rate_None.append(Accuracy_rate_None)
        
        Ave_time_auto = np.mean(tmie_auto, dtype=np.longdouble)
        Ave_time_log2 = np.mean(tmie_log2, dtype=np.longdouble)
        Ave_time_None = np.mean(tmie_None, dtype=np.longdouble)
        Ave_Accuracy_rate_auto = np.mean(accuracy_rate_auto, dtype=np.longdouble)
        Ave_Accuracy_rate_log2 = np.mean(accuracy_rate_log2, dtype=np.longdouble)
        Ave_Accuracy_rate_None = np.mean(accuracy_rate_None, dtype=np.longdouble)
        print("Average time(auto): ", Ave_time_auto)
        print("Average time(log2): ", Ave_time_log2)
        print("Average time(None): ", Ave_time_None)
        print("Average Accuracy rate(auto): ", Ave_Accuracy_rate_auto)
        print("Average Accuracy rate(log2): ", Ave_Accuracy_rate_log2)
        print("Average Accuracy rate(None): ", Ave_Accuracy_rate_None)

        plt.figure(figsize=(10, 8))
        num_range = range(0, 100)
        plt.subplot(2, 1, 1)
        plt.plot(num_range, tmie_auto, ls="-", lw=2, label="max_features=auto")
        plt.plot(num_range, tmie_log2, ls="-.", lw=2, label="max_features=log2")
        plt.plot(num_range, tmie_None, ls="--", lw=2, label="max_features=None")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Running time")
        plt.subplot(2, 1, 2)
        plt.plot(num_range, accuracy_rate_auto, ls="-", lw=2, label="max_features=auto")
        plt.plot(num_range, accuracy_rate_log2, ls="-.", lw=2, label="max_features=log2")
        plt.plot(num_range, accuracy_rate_None, ls="--", lw=2, label="max_features=None")
        plt.legend()
        plt.xlabel("Number of training samples")
        plt.ylabel("Accuracy rate")
        plt.suptitle("Random Forest max_features")
        plt.show()

    else:
        # print("Please input the correct number!")
        return


def all(Num):
    # 定义时间、准确率列表
    Time_KNN = []
    Time_SVM = []
    Time_RF = []
    Time_NB = []
    Time_DT = []
    Time_LR = []
    Accuracy_KNN = []
    Accuracy_SVM = []
    Accuracy_RF = []
    Accuracy_NB = []
    Accuracy_DT = []
    Accuracy_LR = []
    x, y = Load_dataset_digits()

    num = int(Num)  # 测试次数
    for i in range(0, num):
        x_train, x_test, y_train, y_test = Split_dataset(x, y)
        train_SS_x, test_SS_x = Standard_dataset(x_train, x_test)

        k = Choose_K_(train_SS_x, test_SS_x, y_train, y_test)
        start_KNN = time.time()
        y_pred_KNN, accuracy_rate_KNN = KNN(train_SS_x, test_SS_x, y_train, y_test, k)
        costtime_KNN = time.time() - start_KNN
        Time_KNN.append(costtime_KNN)
        Accuracy_KNN.append(accuracy_rate_KNN)

        c, g = Choose_C_gamma(train_SS_x, test_SS_x, y_train, y_test)
        start_SVM = time.time()
        y_pred_SVM, accuracy_rate_SVM = SVM(train_SS_x, test_SS_x, y_train, y_test, c, g)
        costtime_SVM = time.time() - start_SVM
        Time_SVM.append(costtime_SVM)
        Accuracy_SVM.append(accuracy_rate_SVM)

        n = Choose_N_(x_train, x_test, y_train, y_test)
        start_Random_forest = time.time()
        y_pred_RF, accuracy_rate_RF = Random_forest(x_train, x_test, y_train, y_test, n)
        costtime_Random_forest = time.time() - start_Random_forest
        Time_RF.append(costtime_Random_forest)
        Accuracy_RF.append(accuracy_rate_RF)

        start_Naive_bayes = time.time()
        y_pred_NB, accuracy_rate_NB = Naive_bayes(train_SS_x, test_SS_x, y_train, y_test)
        costtime_Naive_bayes = time.time() - start_Naive_bayes
        Time_NB.append(costtime_Naive_bayes)
        Accuracy_NB.append(accuracy_rate_NB)

        start_Decision_tree = time.time()
        y_pred_DT, accuracy_rate_DT = DecisionTree(train_SS_x, test_SS_x, y_train, y_test)
        costtime_Decision_tree = time.time() - start_Decision_tree
        Time_DT.append(costtime_Decision_tree)
        Accuracy_DT.append(accuracy_rate_DT)

        Start_Logistic_Regression = time.time()
        y_pred_LB, accuracy_rate_LB = Logistic_Regression(train_SS_x, test_SS_x, y_train, y_test)
        costtime_Logistic_Regression = time.time() - Start_Logistic_Regression
        Time_LR.append(costtime_Logistic_Regression)
        Accuracy_LR.append(accuracy_rate_LB)

    # 求平均时间
    Ave_time_KNN = np.mean(Time_KNN, dtype=np.longdouble)
    Ave_time_SVM = np.mean(Time_SVM, dtype=np.longdouble)
    Ave_time_RF = np.mean(Time_RF, dtype=np.longdouble)
    Ave_time_NB = np.mean(Time_NB, dtype=np.longdouble)
    Ave_time_DT = np.mean(Time_DT, dtype=np.longdouble)
    Ave_time_LR = np.mean(Time_LR, dtype=np.longdouble)

    # 求平均准确率
    Ave_acc_KNN = np.mean(Accuracy_KNN, dtype=np.longdouble)
    Ave_acc_SVM = np.mean(Accuracy_SVM, dtype=np.longdouble)
    Ave_acc_RF = np.mean(Accuracy_RF, dtype=np.longdouble)
    Ave_acc_NB = np.mean(Accuracy_NB, dtype=np.longdouble)
    Ave_acc_DT = np.mean(Accuracy_DT, dtype=np.longdouble)
    Ave_acc_LR = np.mean(Accuracy_LR, dtype=np.longdouble)

    print("Average accuracy(KNN): ", Ave_acc_KNN)
    print("Average accuracy(SVM): ", Ave_acc_SVM)
    print("Average accuracy(Random forest): ", Ave_acc_RF)
    print("Average accuracy(Naive bayes): ", Ave_acc_NB)
    print("Average accuracy(Decision tree): ", Ave_acc_DT)
    print("Average accuracy(Logistic Regression): ", Ave_acc_LR)

    print("Average running time of KNN: ", Ave_time_KNN)
    print("Average running time of SVM: ", Ave_time_SVM)
    print("Average running time of Random forest: ", Ave_time_RF)
    print("Average running time of Naive bayes: ", Ave_time_NB)
    print("Average running time of Decision tree: ", Ave_time_DT)
    print("Average running time of Logistic Regression: ", Ave_time_LR)

    # 对程序运行时间和准确率进行图表可视化
    plt.figure(figsize=(10, 8))
    num_range = range(0, num)
    plt.subplot(2, 2, 1)
    plt.plot(num_range, Time_KNN, ls="-", lw=2, label="KNN")
    plt.plot(num_range, Time_SVM, ls="-.", lw=2, label="SVM")
    plt.plot(num_range, Time_RF, ls="--", lw=2, label="Random forest")
    plt.legend(loc="upper right")
    plt.xlabel("Number of training samples")
    plt.ylabel("Running time")
    plt.title("Running time of different algorithms")
    plt.subplot(2, 2, 3)
    plt.plot(num_range, Time_NB, ls=":", lw=2, label="Naive bayes")
    plt.plot(num_range, Time_DT, ls="-.", lw=2, label="Decision tree")
    plt.plot(num_range, Time_LR, ls="--", lw=2, label="Logistic Regression")
    plt.legend(loc="upper right")
    plt.xlabel("Number of training samples")
    plt.ylabel("Running time")
    plt.subplot(2, 2, 2)
    plt.plot(num_range, Accuracy_KNN, ls="-", lw=2, label="KNN")
    plt.plot(num_range, Accuracy_SVM, ls="-.", lw=2, label="SVM")
    plt.plot(num_range, Accuracy_RF, ls="--", lw=2, label="Random forest")
    plt.legend(loc="upper right")
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of different algorithms")
    plt.subplot(2, 2, 4)
    plt.plot(num_range, Accuracy_NB, ls=":", lw=2, label="Naive bayes")
    plt.plot(num_range, Accuracy_DT, ls="-", lw=2, label="Decision tree")
    plt.plot(num_range, Accuracy_LR, ls="--", lw=2, label="Logistic Regression")
    plt.legend(loc="upper right")
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.suptitle("Accuracy and running time of different algorithms")
    plt.show()

def single():
    x, y = Load_dataset_digits()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)
    train_SS_x, test_SS_x = Standard_dataset(x_train, x_test)
    print("1. KNN Choose_K")
    print("2. SVM Choose_C")
    print("3. SVM Choose_gamma")
    print("4. Random forest Choose_n_estimators")
    num = input("Please choose: ")
    if num == "1":
        k = Choose_K(train_SS_x, test_SS_x, y_train, y_test)
    elif num == "2":
        c = Choose_C(train_SS_x, test_SS_x, y_train, y_test)
    elif num == "3":
        gamma = Choose_gamma(train_SS_x, test_SS_x, y_train, y_test)
    elif num == "4":
        n_estimators = Choose_N(x_train, x_test, y_train, y_test)
    else:
        # print("Please input the correct number!")
        return

def compare():
    x, y = Load_dataset_digits()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=None)
    train_SS_x, test_SS_x = Standard_dataset(x_train, x_test)
    print("1. Decision tree & Random forest")
    print("2. Logistic Regression(ovr) & SVM(ovr)")
    print("3. MLP(sigmod) & SVM(sigmod)")
    print("4. Algorithm parameter comparison")
    num = input("Please choose: ")
    if num == "1":
        Decision_tree_Random_forest(x_train, x_test, y_train, y_test)
    elif num == "2":
        Logistic_Regression_SVM(train_SS_x, test_SS_x, y_train, y_test)
    elif num == "3":
        MLP_SVM(train_SS_x, test_SS_x, y_train, y_test)
    elif num == "4":
        Algorithm_parameter_comparison(train_SS_x, test_SS_x, y_train, y_test)
    else:
        # print("Please input the correct number!")
        return


def main():
    choose = input("Please select all model tests, single model tests and comparison tests(1-3): ")
    if choose == "1":
        num = eval(input("Please enter the number of experiments: "))
        all(num)
    elif choose == "2":
        single()
    elif choose == "3":
        compare()
    else:
       # print("Please select 1, 2 or 3")
       pass


if __name__ == '__main__':
    main()
