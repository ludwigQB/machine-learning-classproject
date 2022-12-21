import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from PCA import Dimensionality_reduction

data_train=pd.read_csv('../dataset/ECG5000_TRAIN.tsv',sep='\t',header=None)
x=np.array(data_train)[:,1:141]
y=np.array(data_train)[:,0]

data_test=pd.read_csv('../dataset/ECG5000_TEST.tsv',sep='\t',header=None)
x_test=np.array(data_test)[:,1:141]
y_test=np.array(data_test)[:,0]

n_dimension=3
x=Dimensionality_reduction(x,n_dimension)
x_test=Dimensionality_reduction(x_test,n_dimension)

cls=LogisticRegression(solver='liblinear',random_state=0).fit(x,y)
y_pred=cls.predict_proba(x_test)
for i in range(y_pred.shape[0]):
    for j in range(y_pred.shape[1]):
        if y_pred[i,j]==np.max(y_pred[i,:]):
            y_pred[i,j]=j+1
        else:
            y_pred[i,j]=0
y_pred=np.sum(y_pred,axis=1)

y_score=cls.decision_function(x_test)
for i in range(y_pred.shape[0]):
    if y_score[i,4] >10:
        y_test[i] = 5



for j in range(5):
    y_temp=np.copy(y_test)
    for i in range(y_pred.shape[0]):
        if y_temp[i]!=j+1:
            y_temp[i]=-1
        else:
            y_temp[i] = 1
    fpr, tpr, thresolds = metrics.roc_curve(y_temp, y_score[:, j])
    plt.figure()
    ax = plt.gca()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr,tpr)
    plt.legend(["class {}".format(j+1)])
    plt.show()

print('准确率：',metrics.precision_score(y_test,y_pred,average=None,zero_division=0))
print('召回率：',metrics.recall_score(y_test,y_pred,average=None,zero_division=0))
print('混淆矩阵：\n',metrics.confusion_matrix(y_test,y_pred))