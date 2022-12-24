import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import graphviz
from Utils.FunC import Dimensionality_reduction
import matplotlib.pyplot as plt
import seaborn


data_train=pd.read_csv('../dataset/ECG5000_TRAIN.tsv',sep='\t',header=None)
x=np.array(data_train)[:,1:141]
# xmax=np.max(x,axis=0)
# xmin=np.min(x,axis=0)
# x=normalize(x,xmax,xmin,0)
y=np.array(data_train)[:,0]

data_test=pd.read_csv('../dataset/ECG5000_TEST.tsv',sep='\t',header=None)
x_test=np.array(data_test)[:,1:141]
# x_tmax=np.max(x_test,axis=0)
# x_tmin=np.min(x_test,axis=0)
# x_test=normalize(x_test,xmax,xmin,0)
y_test=np.array(data_test)[:,0]

labels=['class 1','class 2','class 3','class 4','class 5']

n_dimension=5
x=Dimensionality_reduction(x,n_dimension)
x_test=Dimensionality_reduction(x_test,n_dimension)

cls=DecisionTreeClassifier(criterion='entropy',random_state=1,min_samples_leaf=5,max_depth=4)
cls.fit(x,y)
y_pred=cls.predict(x_test)
ac=accuracy_score(y_test,y_pred)
print("准确率:%.4lf" % ac)

plt.style.use('ggplot')
plt.subplots(figsize=(30,20))
df=pd.DataFrame(confusion_matrix(y_test,y_pred),
                index=labels,
                columns=labels
                )
seaborn.heatmap(df,annot=True)
plt.show()

dot_data=tree.export_graphviz(cls
                              ,class_names=["1","2","3","4","5"]
                              ,filled=True
                              ,rounded=True)
graph=graphviz.Source(dot_data)
graph.render('../决策树可视化/决策树可视化')
score = cls.score(x_test,y_pred)

print("分类报告:")
print(classification_report(y_test,
                            y_pred,
                            target_names=labels,
                            zero_division=0
                            )
      )
