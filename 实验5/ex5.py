##XGB005t算法案例，全融反欺诈横型
#▣：视型搭建▣
#1,读取数据
import pandas as pd
df = pd.read_excel("信用卡交易数据.xlsx")
df.head()

#2提取特征变量和目标变量
X = df.drop(columns='欺诈标签')
y = df["欺诈标签"]

#3划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#4.模型训练及搭建
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=100, learning_rate=0.05)
clf.fit(x_train, y_train)

#1模型预测及评估
y_pred = clf.predict(x_test)
print(y_pred)

a = pd.DataFrame()  # 创建一个空DataFrame
a["预测值"] = list(y_pred)
a["实际值"] = list(y_test)
a.head()

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)
print(clf.score(x_test, y_test))

y_pred_proba = clf.predict_proba(x_test)
print(y_pred_proba[0:5])  # 查看前5个预测的概率

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:, 1])
print(score)
print(clf.feature_importances_)

features = X.columns  # 获取特征名称
importances = clf.feature_importances_  # 获取特征重要性
importances_df = pd.DataFrame()
importances_df["特征名称"] = features
importances_df["特征重要性"] = importances
importances_df.sort_values("特征重要性", ascending=False)

#模型参数调优
from sklearn.model_selection import GridSearchCV
parameters = {
    "max_depth": [1, 3, 5],
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.05, 0.1, 0.2]
}  # 指定模型中参数的范围

clf = XGBClassifier()  # 构建模型
grid_search = GridSearchCV(clf, parameters, scoring='roc_auc', cv=5)
grid_search.fit(x_train, y_train)  # 传入数据
print(grid_search.best_params_)  # 输出参数的最优值

clf = XGBClassifier(max_depth=1, n_estimators=100, learning_rate=0.05)
clf.fit(x_train, y_train)

y_pred_proba = clf.predict_proba(x_test)
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:, 1])
print(score)