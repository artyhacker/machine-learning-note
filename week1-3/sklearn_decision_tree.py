import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import tree
import graphviz

sData = load_breast_cancer()
X = sData.data
y = sData.target
print(X.shape)
print(y.shape)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

dot_data = tree.export_graphviz(clf, out_file=None,
    feature_names=sData['feature_names'], 
    class_names=sData['target_names'],
    filled=True, rounded=True,
    special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("sklearn_decision_tree")
