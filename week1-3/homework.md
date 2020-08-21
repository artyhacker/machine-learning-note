# Decision Tree

## 1. 获取样本

```python
from sklearn.datasets import load_breast_cancer
import numpy

load_data = load_breast_cancer()
data = load_data.data
target = load_data.target
feature_names = data['feature_names']
# 将结果集作为最后一列
targetInData = numpy.c_[data, target]
```

## 2. 计算样本的信息熵

```python
# 最后一列为结果集
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
```

## 3. 计算属性的信息增益

因为各属性取值为连续值，无法直接计算信息增益，参考C4.5决策树算法，使用二分法分别对每个属性做离散化处理:

1. 对属性取值排序

2. 取每个相邻值的均值作为划分点计算信息增益

   ```python
   def getGain(dataSet, labelIndex):
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
   ```

   

3. 取信息增益最大的点作为划分点，信息增益最大值作为该属性的信息增益

   ```python
   def getGainRatio(dataSet, labelIndex):
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
   ```

   

## 4. 递归建立决策树

1. 对每个属性计算信息增益，选取信息增益最大的属性作为根结点，同时求出其划分点，从划分点分枝；

   ```python
   def chooseBestFeatureToSplit(dataSet):
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
   ```

   

2. 递归地在每个子树上对其他属性再次计算信息增益，建立节点、分枝；

3. 终止条件：子树上所有数据属于同一类(或达到指定深度)。

```python
# 划分数据集
def splitDataSet(dataSet, axis, value, LorR='N'):
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
```



## 5. 5折交叉验证

1. 将样本分为5份，分别使用每1份作为测试集、其他4份作为训练集，得出5组测试/训练集，学习出5组决策树；
3. 最终结果为5组决策树中概率更大的分类

```python
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer
import numpy
# 导入数据
load_data = load_breast_cancer()
data = load_data['data']
target = load_data['target']
target_names = load_data['target_names']
feature_names = load_data['feature_names']
targetInData = numpy.c_[data, target]
# 5折交叉验证
kf = KFold(n_splits=5)
for train, test in kf.split(targetInData):
    _trees = createTree(targetInData, feature_names)
    classLabel = classify(_trees, target_names, feature_names, test)
```

---

> 注1: 仅包含部分代码，完整代码见“my_desicion_tree.py”文件

>  注2: "treePlotter.py"为绘制生成相关方法，基本直接复制了网上相关的代码进行小幅修改