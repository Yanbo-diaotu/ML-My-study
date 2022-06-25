import numpy as np

def loadDataSet():
    '''数据加载函数。这里是一个小例子'''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmatian', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1] # 1 代表侮辱性文字，0 代表正常言论，代表上面 6 个样本的类别
    return postingList, classVec

def createVocabList(dataSet):
    '''
    创建所有文档中出现的不重复词汇列表
    Args:
    dataSet: 所有文档
    Return:包含所有文档的不重复词列表，即词汇表
    '''
    vocabSet = set([])
    # 创建两个集合的并集
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''
    依据词汇表, 将输入文本转化成词集模型词向量
    Args:
    vocabList: 词汇表
    inputSet: 当前输入文档
    Return:
    returnVec: 转换成词向量的文档
    例子：
    vocabList = ['I', 'love', 'python', 'and', 'machine', 'learning']
    inputset = ['python', 'machine', 'learning']
    returnVec = [0, 0, 1, 0, 1, 1]  # 0 1只表示有没有, 不管几次
    长度与词汇表一样长, 出现了的位置为 1, 未出现为 0, 如果词汇表中无该单词则print
    '''
    returnVec = [0]*len(vocabList)  #ket
    # returnVec = np.zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def fit(trainArray, trainCategory):
    '''
    朴素贝叶斯分类器训练函数，求：p(Ci),基于词汇表的 p(w|Ci)
    Args:
    trainArray : 训练样本，即向量化表示后的文档（词条集合）
    trainCategory : 文档中每个词条的列表标注
    Return:
    p0Vect : 属于 0 类别的概率向量(p(w1|C0),p(w2|C0),...,p(wn|C0))
    p1Vect : 属于 1 类别的概率向量(p(w1|C1),p(w2|C1),...,p(wn|C1))
    pAbusive : 属于 1 类别文档的概率
    '''
    numTrainDocs = len(trainArray)
    # 长度为词汇表长度
    numWords = len(trainArray[0])
    # p(ci)，采用拉普拉斯修正，N=2
    pAbusive = (sum(trainCategory) +1)/ (float(numTrainDocs)+2)
    # 由于后期要计算 p(w|Ci)=p(w1|Ci)*p(w2|Ci)*...*p(wn|Ci)，
    # 若 wj 未出现，则 p(wj|Ci)=0,因此 p(w|Ci)=0，这样显然是不对的
    # 故采用拉普拉斯修正，在初始化时，将所有词的出现数初始化为 1，分母即出现词条总数初始化为 2(N_i = 2)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainArray[i]
            p1Denom += sum(trainArray[i])
        else:
            p0Num += trainArray[i]
            p0Denom += sum(trainArray[i])
        # p(wi | c1)
        # 为了避免下溢出（当所有的 p 都很小时，再相乘会得到 0.0，使用 log 则会避免得到0.0）
    p1Vect = np.log2(p1Num / p1Denom)
    # p(wi | c2)
    p0Vect = np.log2(p0Num / p0Denom)
    return pAbusive, p0Vect, p1Vect

def predict(testX, pAbusive, p0Vect, p1Vect):
    '''
    朴素贝叶斯分类器
    Args:
    testX : 待分类的文档向量（已转换成 array）
    p0Vect : p(w|C0)
    p1Vect : p(w|C1)
    pAbusive : p(C1)
    Return:
    1 : 为侮辱性文档 (基于当前文档的 p(w|C1)*p(C1)=log(基于当前文档的p(w|C1))+log(p(C1)))
    0 : 非侮辱性文档 (基于当前文档的 p(w|C0)*p(C0)=log(基于当前文档的p(w|C0))+log(p(C0)))
    '''
    p1 = np.log2(pAbusive) + np.sum(testX * p1Vect)
    p0 = np.log2(1-pAbusive) + np.sum(testX * p0Vect)
    if p1 > p0:
        return 1
    else:
        return 0

if __name__ == "__main__":
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainList = []
    for postDoc in listPosts:
        trainList.append(setOfWords2Vec(myVocabList, postDoc))
    pAbusive, p0Vect, p1Vect = fit(np.array(trainList), np.array(listClasses))
    testEntry1 = ['love', 'my', 'dalmatian']
    thisDoc1 = np.array(setOfWords2Vec(myVocabList, testEntry1))
    print(testEntry1, 'is classified as:', predict(thisDoc1, pAbusive, p0Vect, p1Vect))
    testEntry2 = ['stupid', 'garbage','love', 'my', 'dalmatian']
    thisDoc2 = np.array(setOfWords2Vec(myVocabList, testEntry2))
    print(testEntry2, 'is classified as:', predict(thisDoc2, pAbusive, p0Vect, p1Vect))