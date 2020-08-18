# Decision Tree

## 1. 获取样本

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target
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

### 3.1 连续属性离散化

因为各属性取值为连续值，无法直接计算信息增益，参考C4.5决策树算法，使用二分法对其离散化处理。

理论上应当先对属性取值排序，然后取每个相邻属性的均值作为划分点，计算属性增益并取最大值。

为减少错误、尽快得出结果，第一次先简化计算步骤，直接取中位数作为划分点，后续再按以上步骤对此过程进行优化。
