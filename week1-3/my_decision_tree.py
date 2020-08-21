import math
import numpy
import treePlotter
from sklearn.datasets import load_breast_cancer

# 计算信息熵 样本集最后一列为target
def getEnt(dataSet):
    labelCounts = {}
    for featVec in dataSet:
        if featVec[-1] not in labelCounts.keys():
            labelCounts[featVec[-1]] = 0
        labelCounts[featVec[-1]] += 1 
    result = 0.0
    numEntries = len(dataSet)
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        result -= prob * math.log(prob, 2)
    return result

# 划分数据集
def splitDataSet(dataSet, axis, value, LorR='N'):
    """
    axis: 特征值序号
    value: 划分值
    LorR: L 小于等于value值; R 大于value值
    """
    retDataSet = []
    featVec = []
    if LorR == 'L':
        for featVec in dataSet:
            if float(featVec[axis]) < value:
                retDataSet.append(featVec)
    elif LorR == 'R':
        for featVec in dataSet:
            if float(featVec[axis]) > value:
                retDataSet.append(featVec)
    return retDataSet

def getGain(dataSet, labelIndex):
    """
    type: (list, int) -> float, int
    计算信息增益,返回信息增益值和连续属性的划分点
    dataSet: 数据集
    labelIndex: 特征值索引
    """
    baseEntropy = getEnt(dataSet)  # 计算根节点的信息熵
    featList = [example[labelIndex] for example in dataSet]  # 特征值列表
    uniqueVals = set(featList)  # 该特征包含的所有值
    newEntropy = 0.0

    uniqueValsList = list(uniqueVals)
    sortedUniqueVals = sorted(uniqueValsList)  # 对特征值排序
    listPartition = []
    minEntropy = numpy.inf
    if len(sortedUniqueVals) == 1:  # 如果只有一个值，可以看作只有左子集，没有右子集
        minEntropy = getEnt(dataSet)
    else:
        for j in range(len(sortedUniqueVals) - 1):  # 计算划分点
            partValue = (float(sortedUniqueVals[j]) + float(
                sortedUniqueVals[j + 1])) / 2
            # 对每个划分点，计算信息熵
            dataSetLeft = splitDataSet(dataSet, labelIndex, partValue, 'L')
            dataSetRight = splitDataSet(dataSet, labelIndex, partValue, 'R')
            probLeft = len(dataSetLeft) / len(dataSet)
            probRight = len(dataSetRight) / len(dataSet)
            Entropy = probLeft * getEnt(dataSetLeft) + \
                        probRight * getEnt(dataSetRight)
            if Entropy < minEntropy:  # 取最小的信息熵
                minEntropy = Entropy
    newEntropy = minEntropy
    gain = baseEntropy - newEntropy
    return gain


def getGainRatio(dataSet, labelIndex):
    """
    type: (list, int, int) -> float, int
    计算信息增益率,返回信息增益率和连续属性的划分点
    dataSet: 数据集
    labelIndex: 特征值索引
    """
    baseEntropy = getEnt(dataSet)  # 计算根节点的信息熵
    featList = [example[labelIndex] for example in dataSet]  # 特征值列表
    uniqueVals = set(featList)  # 该特征包含的所有值
    newEntropy = 0.0
    bestPartValuei = None
    IV = 0.0
    
    uniqueValsList = list(uniqueVals)
    sortedUniqueVals = sorted(uniqueValsList)  # 对特征值排序
    listPartition = []
    minEntropy = numpy.inf
    total = len(dataSet)

    if len(sortedUniqueVals) == 1:
        probLeft = 1
        minEntropy = probLeft * getEnt(dataSet)
        IV = -1 * probLeft * math.log(probLeft, 2)
    else:
        for j in range(len(sortedUniqueVals) - 1):  # 计算划分点
            partValue = (float(sortedUniqueVals[j]) + float(
                sortedUniqueVals[j + 1])) / 2
            # 对每个划分点，计算信息熵
            dataSetLeft = splitDataSet(dataSet, labelIndex, partValue, 'L')
            dataSetRight = splitDataSet(dataSet, labelIndex, partValue, 'R')
            totalWeightLeft = len(dataSetLeft)
            totalWeightRight = len(dataSetRight)
            probLeft = totalWeightLeft / total
            probRight = totalWeightRight / total
            Entropy = probLeft * getEnt(
                dataSetLeft) + probRight * getEnt(dataSetRight)
            if Entropy < minEntropy:  # 取最小的信息熵
                minEntropy = Entropy
                bestPartValuei = partValue
                probLeft1 = totalWeightLeft / total
                probRight1 = totalWeightRight / total
                IV += -1 * (probLeft1 * math.log(probLeft1, 2) + probRight1 * math.log(probRight1, 2))

    newEntropy = minEntropy
    gain = baseEntropy - newEntropy
    if IV == 0.0:  # 如果属性只有一个值，IV为0，为避免除数为0，给个很小的值
        IV = 0.0000000001
    gainRatio = gain / IV
    return gainRatio, bestPartValuei


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    """
    type: (list, int) -> int, float
    :param dataSet: 样本集
    :return: 最佳划分属性的索引和连续属性的划分值
    """
    numFeatures = len(dataSet[0]) - 1  # 特征数
    bestInfoGainRatio = 0.0
    bestFeature = -1
    bestPartValue = None  # 连续的特征值，最佳划分值
    gainSum = 0.0
    gainAvg = 0.0
    for i in range(numFeatures):  # 对每个特征循环
        infoGain = getGain(dataSet, i)
        gainSum += infoGain
    gainAvg = gainSum / numFeatures
    for i in range(numFeatures):  # 对每个特征循环
        infoGainRatio, bestPartValuei = getGainRatio(dataSet, i)
        infoGain = getGain(dataSet, i)
        if infoGainRatio > bestInfoGainRatio and infoGain > gainAvg:  # 取信息增益高于平均增益且信息增益率最大的特征
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
            bestPartValue = bestPartValuei
    return bestFeature, bestPartValue


# 通过排序返回出现次数最多的类别
def majorityCnt(classList):
    classCount = {}
    for i in range(len(classList)):
        if classList[i] not in classCount.keys():
            classCount[classList[i]] = 0.0
        classCount[classList[i]] += 1.0

    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    if len(sortedClassCount) == 1:
        return (sortedClassCount[0][0],sortedClassCount[0][1],0.0)
    return (sortedClassCount[0][0], sortedClassCount[0][1], sortedClassCount[1][1])


# 创建树, 样本集 特征名称
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 类别向量
    if classList.count(classList[0]) == len(classList):  # 如果只有一个类别，返回
        totalWeiht = len(dataSet)
        return (classList[0], round(totalWeiht,1),0.0)
    if len(dataSet[0]) == 1:  # 如果所有特征都被遍历完了，返回出现次数最多的类别
        return majorityCnt(classList)
    bestFeat, bestPartValue = chooseBestFeatureToSplit(dataSet)  # 最优分类特征的索引
    if bestFeat == -1:  # 如果无法选出最优分类特征，返回出现次数最多的类别
        return majorityCnt(classList)
    
    bestFeatLabel = labels[bestFeat] + '<' + str(round(bestPartValue, 3))
    myTree = {bestFeatLabel: {}}
    subLabels = labels[:]
    # 构建左子树
    valueLeft = 'Y'
    myTree[bestFeatLabel][valueLeft] = createTree(
        splitDataSet(dataSet, bestFeat, bestPartValue, 'L'), subLabels)
    # 构建右子树
    valueRight = 'N'
    myTree[bestFeatLabel][valueRight] = createTree(
        splitDataSet(dataSet, bestFeat, bestPartValue, 'R'), subLabels)
    return myTree


# 测试算法
def classify(inputTree, classList, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]  # 根节点
    firstLabel = firstStr
    lessIndex = str(firstStr).find('<')
    if lessIndex > -1:  # 如果是连续型的特征
        firstLabel = str(firstStr)[:lessIndex]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstLabel)  # 跟节点对应的特征
    classLabel = {}
    for classI in classList:
        classLabel[classI] = 0.0
    for key in secondDict.keys():  # 对每个分支循环
        partValue = float(str(firstStr)[lessIndex + 1:])
        if testVec[featIndex] == 'N':  # 如果测试样本的属性值缺失，则对每个分支的结果加和
            # 进入左子树
            if type(secondDict[key]).__name__ == 'dict':  # 该分支不是叶子节点，递归
                classLabelSub = classify(secondDict[key], classList, featLabels, testVec)
                for classKey in classLabel.keys():
                    classLabel[classKey] += classLabelSub[classKey]
            else:  # 如果是叶子， 返回结果
                for classKey in classLabel.keys():
                    if classKey == secondDict[key][0]:
                        classLabel[classKey] += secondDict[key][1]
                    else:
                        classLabel[classKey] += secondDict[key][2]
        elif float(testVec[featIndex]) <= partValue and key == 'Y':  # 进入左子树
            if type(secondDict['Y']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                classLabelSub = classify(secondDict['Y'], classList, featLabels, testVec)
                for classKey in classLabel.keys():
                    classLabel[classKey] += classLabelSub[classKey]
            else:  # 如果是叶子， 返回结果
                for classKey in classLabel.keys():
                    if classKey == secondDict[key][0]:
                        classLabel[classKey] += secondDict['Y'][1]
                    else:
                        classLabel[classKey] += secondDict['Y'][2]
        elif float(testVec[featIndex]) > partValue and key == 'N':
            if type(secondDict['N']).__name__ == 'dict':  # 该分支不是叶子节点，递归
                classLabelSub = classify(secondDict['N'], classList, featLabels, testVec)
                for classKey in classLabel.keys():
                    classLabel[classKey] += classLabelSub[classKey]
            else:  # 如果是叶子， 返回结果
                for classKey in classLabel.keys():
                    if classKey == secondDict[key][0]:
                        classLabel[classKey] += secondDict['N'][1]
                    else:
                        classLabel[classKey] += secondDict['N'][2]

    return classLabel

# 导入数据
load_data = load_breast_cancer()
data = load_data['data']
target = load_data['target']
target_names = load_Data['target_names']
feature_names = load_data['feature_names']
targetInData = numpy.c_[data, target]
# 训练数据，建立决策树
trees = createTree(targetInData, feature_names)
# 绘制决策树
treePlotter.createPlot(trees)

# 5折交叉验证
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5)
# for train, test in kf.split(targetInData):
#     _trees = createTree(targetInData, feature_names)
#     classLabel = classify(_trees, target_names, feature_names, test)