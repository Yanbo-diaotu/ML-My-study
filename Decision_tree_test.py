from matplotlib.pyplot import cla
from numpy import *
import pandas as pd
import operator

def calcEnt(dataSet):
    numSamples = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    Ent = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numSamples
        Ent -= prob * log2(prob)
    return Ent

# 获取属性 axis 取值为 value 的样本
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = []
        for example in dataSet:
            featList.append(example[i])
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# 获取类别数目最多的类
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def loadDataSet(filename):
    dataset = pd.read_csv(filename, sep='\s+')
    print(dataset.head())
    myData = dataset.values.tolist()
    attrs = dataset.iloc[:, :-1].columns.values.tolist()
    attrsGroup = {}
    for i in range(len(attrs)):
        featList = []
        for example in myData:
            featList.append(example[i])
        attrsGroup[attrs[i]] = set(featList)
    return myData, attrsGroup

def createTree(dataSet, attrsGroup):
    classList = []
    for example in dataSet:
        classList.append(example[-1])
    # 情形：当前节点包含的样本全属于同一类型
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 情形：当前属性集为空
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 情形：所有样本在所有属性上的取值相同
    if dataSet.count(dataSet[0][:-1]) == len(dataSet):
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = list(attrsGroup.keys())[bestFeat]
    myTree = {bestFeatLabel:{}}
    uniqueVals = attrsGroup[bestFeatLabel]
    del (attrsGroup[bestFeatLabel])
    for value in uniqueVals:
        subAttrsGroup = attrsGroup.copy()
        retDataSet = splitDataSet(dataSet, bestFeat, value)
        if len(retDataSet) == 0: # 情形：节点包含的样本集为空
            myTree[bestFeatLabel][value] = majorityCnt(classList)
        else:
            myTree[bestFeatLabel][value] = createTree(retDataSet, subAttrsGroup)
    return myTree

#test*******************************************

#test*******************************************

def main():
    myData, attrsGroup = loadDataSet('lenses.txt')
    myTree = createTree(myData, attrsGroup)
    print(myTree)

if __name__ == '__main__':
    main()