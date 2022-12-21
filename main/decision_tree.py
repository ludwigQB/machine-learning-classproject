import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import graphviz
from PCA import Dimensionality_reduction


data_train=pd.read_csv('../dataset/ECG5000_TRAIN.tsv',sep='\t',header=None)
x=np.array(data_train)[:,1:141]
y=np.array(data_train)[:,0]

data_test=pd.read_csv('../dataset/ECG5000_TEST.tsv',sep='\t',header=None)
x_test=np.array(data_test)[:,1:141]
y_test=np.array(data_test)[:,0]

n_dimension=5
x=Dimensionality_reduction(x,n_dimension)
x_test=Dimensionality_reduction(x_test,n_dimension)

cls=DecisionTreeClassifier(random_state=1,min_samples_leaf=5,max_depth=4)
cls.fit(x,y)
y_pred=cls.predict(x_test)

dot_data=tree.export_graphviz(cls
                              ,class_names=["1","2","3","4","5"]
                              ,filled=True
                              ,rounded=True)
graph=graphviz.Source(dot_data)
graph.render('../决策树可视化/决策树可视化')
score = cls.score(x_test,y_pred)

print('准确率：',metrics.precision_score(y_test,y_pred,average=None,zero_division=0))
print('召回率：',metrics.recall_score(y_test,y_pred,average=None,zero_division=0))
print('混淆矩阵：\n',confusion_matrix(y_test,y_pred))