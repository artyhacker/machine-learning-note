# Decision Tree

## 1. 获取样本

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target
names = data['feature_names']
```

## 2. 计算样本的信息熵

```python
def getEntropy(y):
  labelCount = {}
  for i in y:
    if i not in labelCount.keys():
      labelCount[i] = 1
    else:
      labelCount[i] += 1
  # print(labelCount) 
  # {0: 212, 1: 357}
  S = len(y)
  result = 0
  for k in labelCount:
    Pk = float(labelCount[k]) / S
    result -= Pk * math.log(Pk, 2)
  # print(result)
  # 0.9526351224018599
  return result
```

## 3. 计算属性的信息增益

因为各属性取值为连续值，无法直接计算信息增益，参考C4.5决策树算法，使用二分法分别对每个属性做离散化处理:

1. 对属性取值排序
2. 取每个相邻值的均值作为划分点计算信息增益
3. 取信息增益最大的点作为划分点，信息增益最大值作为该属性的信息增益

## 4. 递归建立决策树

1. 对每个属性计算信息增益，选取信息增益最大的属性作为根结点，同时求出其划分点，从划分点分枝；
2. 递归地在每个子树上对其他属性再次计算信息增益，建立节点、分枝；
3. 终止条件：子树上所有数据属于同一类(或达到指定深度)。

## 5. 5折交叉验证

1. 将样本随机(分层采样)分为5份；
2. 分别使用每1份作为测试集、其他4份作为训练集，得出5组测试/训练集，学习出5组决策树；
3. 最终结果为5组决策树中概率更大的分类