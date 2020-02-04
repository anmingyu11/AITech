> 导语：完成了数据处理，接下来就可利用平台集成的各算法进行模型训练和模型预测啦。本文将详细介绍“模型训练”、“模型预测”两大模块操作、原理。

模型训练和模型预测是AI策略区别于传统量化策略的核心，我们通过模型训练模块利用训练集因子和标注数据构建一个模型，并通过模型预测模型将预测集的因子数据输入模型进行预测。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/b/3/b3972241f4c6a430f77651bbda44d20facfa189a.jpeg)

在模块列表的 机器学习 、 深度学习 下可找到众多AI算法，可根据需要选择对应算法做训练和预测。

本文以BigQuant专为量化开发的有监督性机器学习算法StockRanker为例，用 StockRanker训练 模块来训练模型，用 StockRanker预测 模块来做出股票预测。

## 一、操作流程

#### 第一步：在模块列表搜索框中输入“StockRanker”，在 机器学习 下找到 StockRanker训练 、 StockRanekr预测 模块并拖入画布。

![](https://cdn.bigquant.com/community/uploads/default/optimized/3X/7/e/7ec35470b94d1013fcf0ae97dc291d38efebd1e0_1_1380x678.png)

#### 第二步：将训练集的 缺失数据处理 和 输入特征列表 连接至 StockRanker训练 模块。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/3/c/3cf271e3f67955e40def4478ed61d803cee9b217.png)

#### 第三步：选中 StockRanker训练 模块，保持默认设置，右键 运行选中模块 。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/a/9/a92a83ccf2a9522ec57e72bb3d3f7a27c26ad5ad.png)

StockRanker算法原理、模块属性参数含义，请继续查看下文。

右键 查看结果1 ，可观察训练好的模型（StockRanker为决策树）样子。

如下图所示，可以看到有20个决策树，每个节点表示决策树的分支条件，每个分支都会依据各自的条件根据预测集中每只股票的因子数据进行打分。最终将所有决策树的打分结果汇总得到每只股票的得分和排序。

![](https://cdn.bigquant.com/community/uploads/default/optimized/3X/a/e/ae35508a15fefbe2da803cb40374b83e3fc81469_1_1380x820.png)

再右键 查看结果2 ，可观察各因子对此模型的贡献重要度分值(gain)。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/1/a/1a156ec65edc41a8c3ef4a5999327bfc7949578d.png)

#### 第四步：将训练好的 StockRanker训练 和测试集的 缺失数据处理 模块连接至 StockRanker预测 。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/d/1/d16d03289a1c3d0fc11cef307228787fb3d229e6.png)

#### 第五步：选中 StockRanker预测 ， 保持默认设置，右键 运行选中模块

![](https://cdn.bigquant.com/community/uploads/default/original/3X/8/c/8c99444ab322dbb5698ea5952f6347e8499168fa.png)

右键 查看结果1 ，可观察每日各股票的得分、排序，即每日最值得买入的股票排序。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/0/d/0dcdbdc248749c5492685af16da561e2ef5941d6.png)

## StockRanker算法介绍：

StockRanker 算法专为量化而生，核心思想是排序学习和梯度提升树。 如下图这个模型有20棵决策树组成，每棵决策树最多有30个叶节点。给定一个样本，每个决策树会对样本打分（分数为样本根据判定条件到达的叶节点的值）；最后的分数是所有决策树打分的总和。

决策树的结构、判定条件和叶节点的分数等等都是由算法在训练数据上学习出来的。将测试集数据喂给训练好的决策树，则可得到测试集数据数据在该模型上的分数，再根据分数形成股票排序、回归、分类预测，指导后续的买入卖出。

![](https://cdn.bigquant.com/community/uploads/default/optimized/3X/6/0/60284ea338a22cbeac3a3a238f2056936d3af20b_1_1380x774.jpeg)

1. 输入数据1：训练数据，如本例中由**缺失数据处理**模块输出的含因子数据、标注数据的训练数据。
2. 输入数据2：因子列表。
3. 学习算法：StockRanker下包含排序、回归、二分类、logloss四种优化算法，默认为排序。
4. 叶节点数量：如上图决策树中每棵树最大叶节点数量。一般情况下，叶子节点越多，则模型越复杂，表达能力越强，过拟合的可能性也越高；默认值是30。
5. 每叶节点最小样本：每个叶节点最少需要的样本数量，一般值越大，泛化性性越好；默认值是1000。
6. 树的数量：如上图决策树中树的数量。一般情况下，树越多，则模型越复杂，表达能力越强，过拟合的可能性也越高；默认值是20。
7. 学习率：学习率如果太大，可能会使结果越过最优值，如果太小学习会很慢；默认值是0.1。
8. 特征值离散化数量：离散化即在不改变因子（即特征）数据相对大小的前提下，对数据进行相应的映射，减小复杂度。离散化数量越大，数据复杂度越高，过拟合的可能性也越高；默认值是1023。
9. 特征使用率：在构建每一颗树时，每个特征（即因子）被使用的概率，如果为1，则每棵树都会使用所有特征；默认值是1。
10. 输出数据1：训练好的模型
11. 输出数据2：各因子对模型的贡献度

> 小结：至此，我们完成了 模型训练 和 股票预测 ，接下来就可利用预测好的股票进行买入卖出，计算并评估对应的收益、风险。
