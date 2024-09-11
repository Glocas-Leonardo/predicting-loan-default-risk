# 项目简介
在项目中，我们将使用 pySpark 中的 MLlib 构建一个端到端的机器学习模型解决类别不平衡的二元分类问题。数据集来自 kaggle 上的[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/overview)竞赛。本次竞赛的目的是根据从每个申请人收集的数据来确定贷款申请人是否有能力偿还贷款。

# 项目内容
## 数据预处理
先简单了解下数据集
![overview](https://github.com/Glocas-Leonardo/Photo/blob/a2db8cf63e09e6cdb64437b49b15679a1b9bf86e/overviewdf.png)

目标变量的分布情况
![label](https://github.com/Glocas-Leonardo/Photo/blob/a2db8cf63e09e6cdb64437b49b15679a1b9bf86e/lable.png)
目标变量为 0（能够偿还贷款的申请人）或 1（无法偿还贷款的申请人）。可以看到目标标签高度不平衡，分配率接近 0.91 到 0.09，其中 0.91 是能够偿还贷款的申请人的比例，0.09 是无法偿还贷款的申请人的比例。

接下来，我们检查有多少个 Categorical  和 Numerical features, 并构建一个函数来输出有关数据集中缺失值的基本信息。然后我们将用每列的平均值填充 Numerical 缺失值，用每列的众数填充 Categorical 缺失值。

在处理完缺失值后，我们需要处理目标变量高度不平衡的问题。在本项目中采取的是加权的方式，给多数类（目标变量为0）赋予 较小的权重（0.09） 的权重，少数类（目标变量为 1 ）赋予较大的权重（0.91），有效地调整模型对不同类别的关注度。

添加权重列之后的结果如下：
![afterweight](https://github.com/Glocas-Leonardo/Photo/blob/78b6c00fb76a4912f8569045573b4bbec5c77c3e/afterweight.png)

## 特征工程
在这一步中，首先我们应用 **StringIndexer()** 为 categorical 列中的每个类别分配索引，用**OneHotEncoderEstimator()** 将 categorical 列转换为 one-hot 编码向量，用 **VectorAssembler()** 将所有的 one-hot 向量和 numerical 特征组合成一个最终的特征向量列 "features"。之后将这个特征组合步骤添加到流水线（stages）中，并以此创建一个 Pipeline。然后用原数据（经数据预处理后的）拟合 Pipeline, 生成一个 PipelineModel, 再将拟合后的的模型应用于原数据，生成一个新的数据集，其中包括编码后的 categorical 特征和组合后的特征向量列。

以下是经过特征工程后我们的新数据集的样子：
![new_df](https://github.com/Glocas-Leonardo/Photo/blob/6ccb979db5ce8d4539bc99d62378f4765453add9/new_df.png)

## 模型训练和超参数调整
为了进行训练，我们首先将数据集分成训练集和测试集。然后我们开始使用逻辑回归进行训练，因为它在二元分类问题中表现良好。使用 PySpark 机器学习库中的 LogisticRegression 类初始化模型，将模型拟合到训练集上并绘制训练数据的 ROC 曲线，以查看逻辑回归的表现。
![ROC](https://github.com/Glocas-Leonardo/Photo/blob/fe7555cacde790b9a016552d3338f3dd050bb484/ROC.png)
之后使用 BinaryClassificationEvaluator 对逻辑回归模型在测试集上的预测结果进行评估，并计算其 ROC 曲线下面积（AUC）：
>Test_SET Area Under ROC：0.7111434396856681

对于逻辑回归来说，0.711 并不是一个非常糟糕的结果。接下来我们尝试另一个模型，梯度提升树（GBT），整体的流程与逻辑回归类似，初始化模型 --> 拟合 --> 对测试集进行预测，最终得到的结果：
>Test_SET Area Under ROC: 0.7322019340889893

使用 GBT，我们能够获得更好的结果，0.732。作为最后的策略，我们将使用网格搜索实现超参数调整，然后运行交叉验证以更好地提高 GBT 的性能。此处使用 ParamGridBuilder 和 CrossValidator 类，前者用于构建网格搜索的参数组合，帮助寻找最优的超参数组合供交叉验证使用；后者用于执行交叉验证，自动测试不同的超参数组合，并选出表现最佳的模型。之后使用用最佳的 GBT 模型对测试集进行预测并评估模型在测试集上的表现：
>CV_GBT (Area Under ROC) = 0.7368288195372332

显然，结果有所改善，同时也意味着我们仍然可以尝试超参数调整，看看是否可以进一步改善结果。

# 总结
本项目利用pySpark的MLlib库，针对Kaggle上的Home Credit Default Risk数据集，成功构建了一个处理类别不平衡的二元分类模型。通过数据预处理解决了缺失值和目标变量不平衡问题，采用加权策略优化模型关注度。特征工程阶段，利用StringIndexer、OneHotEncoderEstimator和VectorAssembler有效转换并组合特征。模型训练阶段，通过逻辑回归和梯度提升树（GBT）对比，发现GBT表现更佳。进一步通过网格搜索和交叉验证优化GBT模型，最终模型在测试集上的AUC值达到0.7368，展现了良好的分类性能，但仍存在进一步提升的空间。