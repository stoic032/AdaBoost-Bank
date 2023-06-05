import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

"""
###################数据处理部分##########################
"""
# 从文件中加载数据集
filepath = r'..\data\train'
filename = 'train'
# 在train目录下从每个表中随机选择十分之一的数据合并
data = pd.DataFrame()
for i in range(1, len(os.listdir(filepath)) + 1):
    df = pd.read_csv(rf'{filepath}\{filename}_{i}.csv', dtype={'GENDER': 'str'})
    df = df.sample(frac=0.1, random_state=42)
    data = pd.concat([data, df], ignore_index=True)

# 数据处理
# 第一列CUST_ID是客户号,第三列IDF_TYP_CD是证件类型,第五列bad_good是是否逾期的结果
data = data.drop(['CUST_ID', 'IDF_TYP_CD', 'GENDER'], axis=1)
# 将所有的N/Y替换为0/1
for column in data.columns:
    # 如果为字符串object
    if data[column].dtype == 'object':
        data[column] = data[column].replace({'N': 0, 'Y': 1})
X = data.drop(['bad_good'], axis=1)
Y = data.loc[:, 'bad_good']

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""
###################训练模型########################
"""
# 迭代m次
m = 10

# 创建AdaBoost分类器，并指定基分类器为朴素贝叶斯
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=m,
    algorithm='SAMME'
)

# 拟合模型
adaboost.fit(x_train, y_train)

# 在测试集上进行预测
y_pred = adaboost.predict(x_test)


# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("测试的准确率为:", accuracy)

"""
####################对test数据进行预测######################
"""
# 从文件中加载数据集
TESTDATA = pd.read_csv(r'..\test.csv', dtype={'GENDER': 'str'})

# 将所有的N/Y替换为0/1
for column in TESTDATA.columns:
    # 如果为字符串object
    if TESTDATA[column].dtype == 'object':
        TESTDATA[column] = TESTDATA[column].replace({'N': 0, 'Y': 1})

TESTDATA_CUST_ID = TESTDATA.loc[:, 'CUST_ID']
TESTDATA_X = TESTDATA.drop(['CUST_ID', 'IDF_TYP_CD', 'GENDER'], axis=1)
TESTDATA_bad_good = adaboost.predict(TESTDATA_X)
filename = r'..\data\submission.csv'
submission = pd.DataFrame({'CUST_ID': TESTDATA_CUST_ID, 'bad_good': TESTDATA_bad_good})
submission.to_csv(filename, index=False)
