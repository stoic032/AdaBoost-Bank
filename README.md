# 机器学习课程设计

数据来源：datafountain竞赛平台

**本设计方案采用了基于AdaBoost算法的集成学习方法，并选择决策树作为基学习器。**
**模型训练：**

采用AdaBoost算法，以决策树作为基学习器，构建集成模型。AdaBoost算法会迭代多次，在每一轮迭代中根据上一轮的分类结果调整样本权重，以提高模型对难分类样本的分类能力。
**模型评估：**

使用交叉验证等方法对模型进行评估，通过计算准确率、召回率、F1值等指标来评估模型的性能。
