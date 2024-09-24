import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# 1. 数据读取
df = pd.read_excel('产品评价.xlsx')

# 确保评价列是数字类型
df['评价'] = df['评价'].astype(float)
df.head()

# 2. 直接使用评价列的数字数据
X = df[['评价']].values
print(X)

# 3. 目标变量提取
y = df["评价"]
y.head()

# 4. 神经网络模型的搭建与使用
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

mlp = MLPClassifier()
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
print(y_pred)

a = pd.DataFrame()  # 创建一个空DataFrame
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
a.head()

score = accuracy_score(y_pred, y_test)
print(score)
print(mlp.score(x_test, y_test))

comment = input('请输入您对本商品的评价（数字）：')
comment = [[float(comment)]]
print(comment)
y_pred = mlp.predict(comment)
print(y_pred)

nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)
y_pred = nb_clf.predict(x_test)
print(y_pred)

score = accuracy_score(y_pred, y_test)
print(score)