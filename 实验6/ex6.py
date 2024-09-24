# 读取数据
import pandas as pd
df = pd.read_excel('新闻.xlsx')
df.head()

import jieba
words = []
for _, row in df.iterrows():
    words.append(' '.join(jieba.cut(row['标题'])))
print(words[0:3])

# 并所有特征文本量化并显示
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(words)  # 将分词后的内容文本向量化
words_bag = vect.get_feature_names_out()  # 第二种查看词袋的方法
df = pd.DataFrame(X.toarray(), columns=words_bag)

# 显示pandas中DataFrame所有行，所有列
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.head()

# 通过KMeans算法进行聚类分析
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=10, random_state=123)
k_data = kms.fit_predict(df)
print(k_data)

import numpy as np
words_ary = np.array(words)
print(words_ary[k_data == 1])  # 可以把数字1改成其他数字看看效果

# 通过DBSCAN算法进行聚类分析
from sklearn.cluster import DBSCAN
dbs = DBSCAN(eps=1, min_samples=3)
d_data = dbs.fit_predict(df)
print(d_data)

# 模型优化（利用余弦相似度进行优化）
# 余弦相似度基本概念
words_test = ['想去华能信托', '华能信托很好想去', '华能信托很好想去华能信托很好想去']
vect = CountVectorizer()
X_test = vect.fit_transform(words_test)  # 将分词后的内容文本向量化
X_test = X_test.toarray()
words_bag2 = vect.get_feature_names_out()  # 第二种查看词袋的方法
df_test = pd.DataFrame(X_test, columns=words_bag2)
df_test.head()

# 通过numpy计算欧式距离
dist = np.linalg.norm(df_test.iloc[0] - df_test.iloc[1])
print(dist)

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarities = cosine_similarity(df_test)
print(cosine_similarities)

# 余弦相似度实战·模型优化
cosine_similarities = cosine_similarity(df)
print(cosine_similarities)

kms = KMeans(n_clusters=10, random_state=123)
k_data = kms.fit_predict(cosine_similarities)
print(k_data)
print(k_data == 3)
print(words[0:3])
words_ary = np.array(words)
print(words_ary[0:3])
print(words_ary[k_data == 3])  # 可以把数字3改成其他数字看看效果