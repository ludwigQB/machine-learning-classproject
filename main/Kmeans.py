import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from PCA import Dimensionality_reduction
import matplotlib.pyplot as plt

data_train=pd.read_csv('../dataset/ECG5000_TRAIN.tsv',sep='\t',header=None)
x=np.array(data_train)[:,1:141]
y=np.array(data_train)[:,0]

data_test=pd.read_csv('../dataset/ECG5000_TEST.tsv',sep='\t',header=None)
x_test=np.array(data_test)[:,1:141]
y_test=np.array(data_test)[:,0]

# cls=DBSCAN(min_samples=1,eps=4) ##DBSCAN中噪声点过多
cls=KMeans(n_clusters=3,n_init=10,random_state=1) ##4类与5类样本过少，若选择五个聚类中心则聚类效果较差
cls.fit(x)
y_pred=cls.fit_predict(x_test)

##画图对比测试样本聚类效果
n_dimension=2
x_test=Dimensionality_reduction(x_test,n_dimension)
plt.figure()
plt.subplot(1,2,1)
for target in y_test:
    plt.scatter(x_test[:,0][y_test==target],x_test[:,1][y_test==target],linewidths=0.01)
plt.title('label known')

plt.subplot(1,2,2)
for target in y_pred:
    plt.scatter(x_test[:,0][y_pred==target],x_test[:,1][y_pred==target])
plt.title('label unknown')
plt.show()
