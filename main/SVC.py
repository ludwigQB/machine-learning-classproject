import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC

data_train = pd.read_csv('../dataset/ECG5000_TRAIN.tsv', sep='\t', header=None)
x = np.array(data_train)[:, 1:141]
y = np.array(data_train)[:, 0]

data_test = pd.read_csv('../dataset/ECG5000_TEST.tsv', sep='\t', header=None)
x_test = np.array(data_test)[:, 1:141]
y_test = np.array(data_test)[:, 0]

labels = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5']

# n_dimension=50
# x=Dimensionality_reduction(x,n_dimension)
# x_test=Dimensionality_reduction(x_test,n_dimension)

cls = SVC(random_state=5000, max_iter=50000).fit(x, y)
y_pred = cls.predict(x_test)
y_score = cls.decision_function(x_test)

for j in range(5):
    y_temp = np.copy(y_test)
    for i in range(y_pred.shape[0]):
        if y_temp[i] != j + 1:
            y_temp[i] = -1
        else:
            y_temp[i] = 1
    fpr, tpr, thr = roc_curve(y_temp, y_score[:, j])
    plt.figure()
    ax = plt.gca()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr)
    plt.legend(["class {}".format(j + 1)])
    plt.show()

ac = accuracy_score(y_test, y_pred)
print("准确率:%.4lf" % ac)

plt.figure()
df = pd.DataFrame(confusion_matrix(y_test, y_pred),
                  index=labels,
                  columns=labels
                  )
seaborn.heatmap(df, annot=True)
plt.show()

print("分类报告:")
print(classification_report(y_test,
                            y_pred,
                            target_names=labels,
                            zero_division=0
                            )
      )

train_sizes, train_scores, test_scores = learning_curve(cls, x, y, cv=2, n_jobs=-1)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_ylim((0, 1.1))
ax.set_xlabel("training examples")
ax.set_ylabel("score")
ax.grid()
ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='train score')
ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='test score')
ax.legend(loc='best')
plt.show()
