import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

from Utils.FunC import Dimensionality_reduction

data_train = pd.read_csv('../dataset/ECG5000_TRAIN.tsv', sep='\t', header=None)
x = np.array(data_train)[:, 1:141]
y = np.array(data_train)[:, 0]

data_test = pd.read_csv('../dataset/ECG5000_TEST.tsv', sep='\t', header=None)
x_test = np.array(data_test)[:, 1:141]
y_test = np.array(data_test)[:, 0]

labels = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5']

# n_dimension = 3
# x = Dimensionality_reduction(x, n_dimension)
# x_test = Dimensionality_reduction(x_test, n_dimension)

cls = LogisticRegression(solver='liblinear', random_state=0)
cls.fit(x, y)
y_pred = cls.predict_proba(x_test)
for i in range(y_pred.shape[0]):
    for j in range(y_pred.shape[1]):
        if y_pred[i, j] == np.max(y_pred[i, :]):
            y_pred[i, j] = j + 1
        else:
            y_pred[i, j] = 0
y_pred = np.sum(y_pred, axis=1)

ac = accuracy_score(y_test, y_pred)
print("准确率:%.4lf" % ac)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
disp.plot()
plt.savefig('../结果可视化/逻辑回归/混淆矩阵.png')
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
plt.savefig('../结果可视化/逻辑回归/学习曲线.png')
plt.show()
