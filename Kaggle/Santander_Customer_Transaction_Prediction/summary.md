> https://www.kaggle.com/c/santander-customer-transaction-prediction/overview

# Santander Customer Transaction Prediction

Can you identify who will make a transaction?

1) 数据不平衡

2) 特征匿名

3) 偏度和峰度

4) 特征工程

5) 部分依赖图(PDP)


## 1) 数据不平衡

## 2) 特征匿名

### 大量EDA分析

#### 分别对应t0,t1,train,test ,eda 每个样本的所有特征值的(mean,std,min,max,skew,kurtosis,median)

```
t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train set")
sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend()
plt.show()
```

#### 分别对应t0,t1,train,test ,eda 每个特征的(mean,std,min,max,skew,kurtosis，median)

```
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train set")
sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```

#### 对某个特征A以及其对应的类别切分成两个数据集，查看数据分布，对数据集Data[A,target==0],Data[A,target==1]


绘制多个特征的对应的分布:

```
def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
        
    plt.show();
```

```
t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)
```

#### 分别绘制训练集和测试集对应特征值得分布

```
features = train_df.columns.values[2:102]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)
```

### 相关系数分析

### 重复值

## 3) 偏度和峰度

[https://zhuanlan.zhihu.com/p/53184516](https://zhuanlan.zhihu.com/p/53184516)

#### 偏度（Skewness）

偏度衡量随机变量概率分布的不对称性，是相对于平均值不对称程度的度量，通过对偏度系数的测量，我们能够判定数据分布的不对称程度以及方向。

偏度的衡量是相对于正态分布来说，正态分布的偏度为0，即若数据分布是对称的，偏度为0。若偏度大于0，则分布右偏，即分布有一条长尾在右；若偏度小于0，则分布为左偏，即分布有一条长尾在左（如下图）；同时偏度的绝对值越大，说明分布的偏移程度越严重。

![](https://pic2.zhimg.com/80/v2-e702671ffdfb6997dd6e23de014e12f5_hd.jpg)

**【注意】数据分布的左偏或右偏，指的是数值拖尾的方向，而不是峰的位置。**

#### 峰度（Kurtosis）

峰度，是研究数据分布陡峭或者平滑的统计量，通过对峰度系数的测量，我们能够判定数据相对于正态分布而言是更陡峭还是更平缓。比如正态分布的峰度为0，均匀分布的峰度为-1.2（平缓），指数分布的峰度6（陡峭）。

- 若峰度 ≈ 0 , 分布的峰态服从正态分布；
- 若峰度 > 0 , 分布的峰态陡峭（高尖）；
- 若峰度 < 0 , 分布的峰态平缓（矮胖）；

![](https://pic2.zhimg.com/80/v2-67887d745da1b7dce1468bfccb451d99_hd.jpg)

### 正态性检验
利用变量的偏度和峰度进行正态性检验时，可以分别计算偏度和峰度的Z评分（Z-score）。

- 偏度Z-score = 偏度值 / 偏度值的标准差
- 峰度Z-score = 峰度值 / 峰度值的标准差
- 
在 α = 0.05 的检验水平下，偏度Z-score和峰度Z-score是否满足假设条件下所限制的变量范围（Z-score在±1.96之间），若都满足则可认为服从正态分布，若一个不满足则认为不服从正态分布。

### 正态性检验的适用条件

样本的增加会减小偏度值和峰度值的标准差，相应的Z-score会变大，最终会拒绝条件假设，会给正确判断样本数据的正态性情况造成一定的干扰。

因此，当样本数据量小于100时，用偏度和峰度来判断样本的正态分布性比较合理。

### SPSS结果分析

![](https://pic4.zhimg.com/80/v2-517b64af3d26ea376af24afeb6738c73_hd.jpg)

上图中可以看出分布的偏度值为0.194(偏度值的标准差0.181)
，则Z-score = 0.194 / 0.181 = 1.072；

峰度值0.373（峰度值标准差0.360），则Z-score = 0.373 / 0.360 = 1.036。

偏度值和峰度值均 ≈ 0,Z-score均在 ± 1.96之间，可认为资料服从正态分布。



## 4) 特征工程

都是连续型特征

将每个样本的所有特征的值求其相关的统计量作为特征

```
%%time
idx = features = train_df.columns.values[2:202]
for df in [test_df, train_df]:
    df['sum'] = df[idx].sum(axis=1)  
    df['min'] = df[idx].min(axis=1)
    df['max'] = df[idx].max(axis=1)
    df['mean'] = df[idx].mean(axis=1)
    df['std'] = df[idx].std(axis=1)
    df['skew'] = df[idx].skew(axis=1)
    df['kurt'] = df[idx].kurtosis(axis=1)
    df['med'] = df[idx].median(axis=1)
```

## 5) 部分依赖图
> 原文：[https://blog.waynehfut.com/2019/03/19/MLExplainbility-3/](https://blog.waynehfut.com/2019/03/19/MLExplainbility-3/)

排列重要性展示了哪些变量**最容易**影响预测结果

而部分依赖图则是展示了特征**如何影响**最终预测结果。

这项手段对于回答下列问题十分有利：

- 在固定住其他房屋相关的特征后，经度和纬度对房价会有什么样的影响？重申这一点，即为在不同位置房子应该如何定价？
- 预测两组人群的健康差异主要是由于饮食习惯还是其他的因素？

如果你对线性回归和逻辑回归模型很熟悉，部分依赖图可以用模型中相似的系数来解释。但是，复杂模型的部分依赖图可以捕获到比简单模型中系数更为复杂的模式。如果你对线性回归和逻辑回归不是很熟悉，也不用担心这些比较。

我们将展示一系列代码，解释这些图的背后涵义，之后审查代码以便于后期创建这些图

```
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='var_82')

# plot it
pdp.pdp_plot(pdp_goals, 'var_82')
plt.show() 
```


就像排列重要性，部分依赖图在模型训练完毕后才进行计算。这些模型在真实数据中调优同时没有以其他任何方式进行人为修改。

在我们之前提到的足球例子中，队伍之间有诸多的不同。如：过了多少个人，射门多少次，记录的进球等。乍一看似乎很难解开其中这些特征对最后结果的影响。

要查看部分图如何分离每个要素的效果，我们首先考虑单列数据。例如，一列数据可能展示了50%的控球率，100次过人，10次射门和1次进球。

我们将使用已经训练好的模型去预测可能的结果（获得"最佳球员"的概率）。但我们反复改变一个变量的值来进行一系列的预测。我们可以预测队伍控球40%时的结果，之后预测控球50%时的结果，以及在60%时的结果，等等。我们追踪控球率（横轴）从小到大变化时的预测结果（纵轴）。

基于上述描述，我们仅使用一列数据。特征之间的相互影响可能会造成单列图异常。因此，我们在原始数据集中的多行不断的重复这个心理实验，之后我们在垂直坐标上画出平均预测结果。

### 如何生效

就像排列重要性，**部分依赖图在模型训练完毕后才进行计算**。这些模型在真实数据中调优同时没有以其他任何方式进行人为修改。

在我们之前提到的足球例子中，队伍之间有诸多的不同。如：过了多少个人，射门多少次，记录的进球等。乍一看似乎很难解开其中这些特征对最后结果的影响。

要查看部分图如何分离每个要素的效果，我们首先考虑单列数据。例如，一列数据可能展示了50%的控球率，100次过人，10次射门和1次进球。

**我们将使用已经训练好的模型去预测可能的结果（获得"最佳球员"的概率）。但我们反复改变一个变量的值来进行一系列的预测。我们可以预测队伍控球40%时的结果，之后预测控球50%时的结果，以及在60%时的结果，等等。我们追踪控球率（横轴）从小到大变化时的预测结果（纵轴）。**

基于上述描述，我们仅使用一列数据。特征之间的相互影响可能会造成单列图异常。因此，我们在原始数据集中的多行不断的重复这个心理实验，之后我们在垂直坐标上画出平均预测结果。

### 代码

建立模型仍然不是我们关注的，因此我们不去关注数据探索或者是模型建立的代码。

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # 将"Yes"/"No"字符串转换为二值
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
```

为了便于解释（For the sake of explanation)，我们第一个例子使用你可以在下面看到的决策树。在实践中，你将需要适用于真是世界中更加复杂的模型。

```
from sklearn import tree
import graphviz

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
graphviz.Source(tree_graph)
```

![](https://blog.waynehfut.com/2019/03/19/MLExplainbility-3/p1.png)

理解这棵树：

有孩子节点的节点在顶部展示了分割标准。

底部的值对分别显示了树的该节点中数据符合分割标准的真值和假值的数量

接下来是使用PDPBox library创建部分依赖图的代码

```
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# 创建需要绘图的数据
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature='Goal Scored')

# 绘图
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()
```

![](https://blog.waynehfut.com/2019/03/19/MLExplainbility-3/p2.png)

在解释这个图时，有一些值得注意的点。

- y轴用以解释预测中从基线或最左边值预测的变化
- 蓝色阴影表示了置信度

对于这个特定的图而言，我们可以看出一次进球数能够大大的增加你获得“最佳球员”的机会。但是额外的进球对于预测而言并没有太多的影响。

这里是另一个例图：

```
feature_to_plot = 'Distance Covered (Kms)'
pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()
```

![](https://blog.waynehfut.com/2019/03/19/MLExplainbility-3/p3.png)

这个图似乎过于简单而无法表现出真实现象。实质上是模型太过简单了，你应该可以从上面的决策树发现这个实际上代表了模型的结构（waynehfut注：101.5km为节点）

你可以很容易的比较不同模型的结构和涵义，这里提供了一个随机森林模型的示例图。

```
# 构建随机森林模型
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

pdp_dist = pdp.pdp_isolate(model=rf_model, dataset=val_X, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()
```

![](https://blog.waynehfut.com/2019/03/19/MLExplainbility-3/p4.png)

这个模型认为你如果跑动超过100km的话，更有可能获得最佳球员。虽然跑的更多导致了更低的预测结果。

通常，这个曲线的平滑形状似乎比决策树模型中的阶梯函数更合理。因为这个数据集足够小因此我们需要在解释模型时小心翼翼的处理。

### 2D的部分依赖图

如果你关心特征间的相互影响，那么二维的部分依赖图将会很有用，一个例子可以说明清楚这个是什么。

我们对于这个图仍然使用决策树。他将构建一个非常简单的图，但是你应当可以将这个图与原来的决策树进行匹配。

```
# 与前文的部分依赖图相似，我们使用pdp_interact替换了pdp_isolate,pdp_interact_plot替换了pdp_isolate_plot
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=val_X, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()
```

结果如下：

![](https://blog.waynehfut.com/2019/03/19/MLExplainbility-3/p5.png)

这个图展示了任意进得分和覆盖距离组合可能的预测结果

例如，我们看到最高预测结果出现在至少进了一球并跑动接近100km的时候。如果没有进球，覆盖距离也无关紧要了。你能够通过追踪0进球的决策树来看到这一点吗？

但是，如果他们获得进球，距离会影响预测。确保您可以从2D部分依赖图中看到这一点。你能在决策树中看到这种模式吗？

## 6) SHAP值

### 简介

你已经看到（或使用了）从机器学习中提前常规洞察的相关技术。但是如果你想在一次独立预测中分解模型的话该怎么做呢？

SHAP值(Shapley加法解释的缩写)可以分解预测结果以展示每个值的影响。那么他可以用在什么地方呢？

- 一个模型说银行不应当贷款给某个人，同时法律要求银行需要每次解释拒绝贷款的缘由时。
- 一个健康服务机构想确定哪些因素导致每个患者患有某种疾病的风险，从而他们可以有针对性的健康干预并直接解决掉这些风险因素。

你将在这节课程中使用SHAP值来解释单个预测。在下一节课中，你还将看到它如何与其他模型级的洞察方法结合。

### 如何运作

SHAP值解释了对于给定特征具有某些特定值所产生的影响，并与我们在该特征具有某些基线值时所作的预测进行比较。

一个示例将有利于理解，我们本节仍将使用前文排列重要性和部分依赖图所使用的足球/橄榄球的例子。

在之前的课程中，我们预测了球队是否会有对于获得最佳球员奖。

我们也许会问：

- 如果球队打入了3球会怎样推动预测结果？

其实这个问题可以更加具体，量化的回答可以被重写为：

- 如果球队打入了3球而不是一些基准进球数这一事实将会怎样推动预测结果？

当然，每个队伍都会有很多的特征。因此我们如果回答进球数这个问题，我们也可以为所有其他功能重复此过程。

SHAP值将以保证良好属性的方式执行此操作。当我们进行如下的预测时：

```
sum(SHAP values for all features) = pred_for_team - pred_for_baseline_values
```

也就是说，所有特征的SHAP值是对为什么预测与基线不同的一个求和。这也允许我们将预测分解为下图：

![](https://blog.waynehfut.com/2019/03/23/MLExplainbility-4/p1.png)

那如何理解这些呢？

当我们作出0.7的预测时(即该队有70%概率获得最佳球员，waynehfut注)，与此同时基线值是0.4979。特征值导致预测值的增长的由粉色区域标出，而它的视觉尺寸大小衡量了特征的影响。特征导致的降低效果由蓝色区域标出。影响力最大的值是来自于`Goal Scored为`2。与此同时控球率具有降低预测结果的效果。

如果从粉红条的长度中减去蓝条的长度，则它等于从基值到结果的距离。

这项技术仍有一些复杂性，为了确保基线加上各个独立预测的总和相加的值与预测值相等(听上去似乎不是那么简单)。我们将不会在这里讨论细节，因为这个技术并不重要，欲知详情可以在这个博文中得到解释

### 计算SHAP值的代码

我们将使用优秀的Shap库来计算SHAP值。

例如，我们将重用你已经看过所有数据的模型。

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # 二值化
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
```

我们将查看数据集中单列数据的SHAP值（我们随机选择第五行）。对于其中内容，我们在查看SHAP值之前先看原始预测值。

```
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # 选取第5列数据，如果有必要可以全选
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
my_model.predict_proba(data_for_prediction_array)
```

```
array([[0.3, 0.7]])
```

这个队伍有70%的概率获得这个奖项。

接着我们来关注下单次预测的SHAP值

```
import shap  # 导入计算shap值的包

# 创建可以计算Shap值的对象
explainer = shap.TreeExplainer(my_model)

# 计算单次预测的Shap值
shap_values = explainer.shap_values(data_for_prediction)
```

上述的`shap_values`是一个有两个数组的列对象。第一个数组SHAP值表示负面的输出（无法获得奖），SHAP值的第二个数组是代表了正面的输出（赢得比赛）。我们通常会考虑预测积极结果的预测，因此我们将把所有SHAP值的积极输出提出（使用shap_values[1]）

查看原始数组很麻烦，但是shap包有一个可视化结果的好方法。

```
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
```

![](https://blog.waynehfut.com/2019/03/23/MLExplainbility-4/p2.png)

如果你仔细的查看创建SHAP值的代码，你将会注意到我们参考了`shap.TreeExplainer(my_model)` 中的树。但是SHAP包已经解释了模型的每种类型。

- `shap.DeepExplainer`在深度模型中有效果
- `shap.KernelExplainer`在所有模型中都有效，但他比其他的解释器慢了一些，且他提供了似值而不是准确值

下面提供了一个使用`KernelExplainer`的例子以获取相似的结果。结果值不是完全一致，因为`KernelExplainer`给出了一个近似的结果。但是结果值表示的意思是一致的。

```
# 使用SHAP核解释测试集预测结果
k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
```

![](https://blog.waynehfut.com/2019/03/23/MLExplainbility-4/p3.png)
