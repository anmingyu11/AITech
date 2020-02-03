> https://www.kaggle.com/c/pubg-finish-placement-prediction

# PUBG Finish Placement Prediction (Kernels Only)

Can you predict the battle royale finish of PUBG Players?

## Data Description

In a PUBG game, up to 100 players start in each match (matchId). Players can be on teams (groupId) which get ranked at the end of the game (winPlacePerc) based on how many other teams are still alive when they are eliminated. In game, players can pick up different munitions, revive downed-but-not-out (knocked) teammates, drive vehicles, swim, run, shoot, and experience all of the consequences -- such as falling too far or running themselves over and eliminating themselves.

You are provided with a large number of anonymized PUBG game stats, formatted so that each row contains one player's post-game stats. The data comes from matches of all types: solos, duos, squads, and custom; there is no guarantee of there being 100 players per match, nor at most 4 player per group.

You must create a model which predicts players' finishing placement based on their final stats, on a scale from 1 (first place) to 0 (last place).

------------------------

File descriptions

- train_V2.csv - the training set
- test_V2.csv - the test set
- sample_submission_V2.csv - a sample submission file in the correct format
- Data fields
- DBNOs - Number of enemy players knocked.
- assists - Number of enemy players this player damaged that were killed by teammates.
- boosts - Number of boost items used.
- damageDealt - Total damage dealt. Note: Self inflicted damage is subtracted.
- headshotKills - Number of enemy players killed with headshots.
- heals - Number of healing items used.
- Id - Player’s Id
- killPlace - Ranking in match of number of enemy players killed.
- killPoints - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
- killStreaks - Max number of enemy players killed in a short amount of time.
- kills - Number of enemy players killed.
- longestKill - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
- matchDuration - Duration of match in seconds.
- matchId - ID to identify match. There are no matches that are in both the training and testing set.
- matchType - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
- rankPoints - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
- revives - Number of times this player revived teammates.
- rideDistance - Total distance traveled in vehicles measured in meters.
- roadKills - Number of kills while in a vehicle.
- swimDistance - Total distance traveled by swimming measured in meters.
- teamKills - Number of times this player killed a teammate.
- vehicleDestroys - Number of vehicles destroyed.
- walkDistance - Total distance traveled on foot measured in meters.
- weaponsAcquired - Number of weapons picked up.
- winPoints - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
- groupId - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
- numGroups - Number of groups we have data for in the match.
- maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
- winPlacePerc - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

----------------------

训练集的样本数量: 400万+ 测试集大约269万+

这个竞赛学到的东西并没有那么多，只是数据处理起来非常的复杂，共涉及到以下几点。


> 1) 特征工程与特征重要性

> 2) 特征工程中的大量聚合操作

> 3) 如何通过eda来检测异常值

> 4) 技巧: `reduce_mem_usage`.

## 1) 特征工程与特征重要性

### 特征重要性

1. Correlation
2. 在简单模型上的训练效果
3. 树模型输出的特征重要性
4. Permutation importance
5. SHAP 值
6. 在复杂模型上获得分数

------------------------

#### 1. Correlation

检查 Correlation 是估计特征重要性的最快方法，但 Correlation 并不能获得特征对模型训练分数的实质性贡献。

#### 2. Score gain on a simple model

我在特征工程中使用了线性回归，因为它又简单又块。如果你只是想看看你添加的 NewFeature 的影响，这就足够了。

1. EDA
2. 建立一个简单的模型
3. 在一个简单的模型上尝试各种特征
4. 建立一个复杂的模型
5. 用较为可靠的的特征训练复杂模型 

在比赛中，反复的用上述方法，就会得到较大的优势。


代码 :

```
def run_experiment(preprocess):
    df = reload()
    df.drop(columns=['matchType'],inplace=True)
    
    df = preprocess(df)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    train, val = train_test_split(df, 0.1)
    
    model = LinearRegression()
    model.fit(train[cols_to_fit],train[target])
    
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true,y_pred)
```

```
def run_experiments(preprocesses):
    results = []
    for preprocess in preprocesses:
        start = time.time()
        score = run_experiment(preprocess)
        execution_time= time.time() - start
        results.append(
            {
                'name' : preprocess.__name__ # 函数对应的名称
                , 'score' : score # 所得分数
                , 'execution time' : f'{round(execution_time,2)}s' # 运行时间
            }
        )
        gc.collect()
        
        return pd.DataFrame(results,columns =['name','score','execution time']).sort_values(by='score')# list [map] 转换成dataframe
```

```
# 下面都是创建新特征的函数
run_experiments([
    original,
    items,
    players_in_team,
    total_distance,
    headshotKills_over_kills,
    killPlace_over_maxPlace,
    walkDistance_over_heals,
    walkDistance_over_kills,
    teamwork
])
```

### 3. Feature importances of Tree models

代码:

```
feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_, cols_to_fit)), columns=['Value','Feature'])

plt.figure(figsize=(10, 6))
sns.barplot(x="Value", y="Feature", data=feature_importance.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
```

There are several options for measuring importance like "split" (How many times the feature is used to split), "gain" (The average training loss reduction gained when using a feature for splitting).

> 几个衡量重要性的选择： 如 split(特征有多少次用于划分) , 'gain' 当用这个特征来划分时损失函数减小多少

However, sometimes it doesn't represent the actual contribution.

> 但有时候这并不代表真正的贡献

To our dismay we see that the feature importance orderings are very different for each of the three options provided by XGBoost!

For the cover method it seems like the capital gain feature is most predictive of income, while for the gain method the relationship status feature dominates all the others.

This should make us very uncomfortable about relying on these measures for reporting feature importance without knowing which method is best.

> 令我们沮丧的是，我们看到XGBoost提供的三个选项的重要特性排序非常不同!
> 对于覆盖法来说，capital gain feature 似乎是最有效的
> 而对于收益法来说，relationship status 特征是最主要的。
> 这应该让我们对依赖这些 meatures 来报告feature 重要性而不知道哪种方法是最好的感到非常不舒服。

<img src="https://cdn-images-1.medium.com/max/1600/1*UEQiHKTnjHJ-swIjcAkRnA.png" width="640">

### 4. Permutation importance

The basic idea is that observing how much the score decreases when a feature is not available; the method is known as “permutation importance” or “Mean Decrease Accuracy (MDA)”.

> 其基本思想是观察当某个特性不可用时，得分下降了多少;该方法被称为“排列重要性”或“平均降低准确性(MDA)”。

The figure shows the importance of each feature.

ELI5 shuffles the target feature instead of removing it to make it useless so that we don't need to re-train the model again. 

That's why it's represented like ± 0.0033 (standard deviation).

> 图中显示了每个feature的重要性。

> ELI5对目标特性进行了shuffles，而不是删除它make it useless，这样我们就不需要重新训练模型了。

> 这就是为什么它表示为±0.0033(标准差)

Removing a feature and see the difference... this is what I was doing above, but more reliable! However, there is room to discuss how to define/measure contribution.

> 删除一个feature，看看有什么不同…这就是我在上面所做的，但是更可靠!但是，还有讨论如何定义/度量贡献的空间。

代码:

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=42).fit(val[cols_to_fit], val[target])
eli5.show_weights(perm, feature_names=list(cols_to_fit))
```

### 5. SHAP values

SHAP是Python开发的一个"模型解释"包，可以解释任何机器学习模型的输出。其名称来源于SHapley Additive exPlanation，在合作博弈论的启发下SHAP构建一个加性的解释模型，所有的特征都视为“贡献者”。对于每个预测样本，模型都产生一个预测值，SHAP value就是该样本中每个特征所分配到的数值。

传统的feature importance只告诉哪个特征重要，但我们并不清楚该特征是怎样影响预测结果的。SHAP value最大的优势是SHAP能对于反映出每一个样本中的特征的影响力，而且还表现出影响的正负性。


#### 5.1 Explainer

在SHAP中进行模型解释需要先创建一个explainer，SHAP支持很多类型的explainer(例如deep, gradient, kernel, linear, tree, sampling)，我们先以tree为例，因为它支持常用的XGB、LGB、CatBoost等树集成算法

#### 5.2 单个prediction的解释

```
# 可视化第一个prediction的解释   如果不想用JS,传入matplotlib=True
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
```

#### 5.3 多个预测的解释

```
shap.force_plot(explainer.expected_value, shap_values, X)
```

#### 5.4 Global Interper

Global可解释性：寻求理解模型的overall structure(总体结构)。这往往比解释单个预测困难得多，因为它涉及到对模型的一般工作原理作出说明，而不仅仅是一个预测。

##### 5.4.1 summary_plot

summary plot 为每个样本绘制其每个特征的SHAP值，这可以更好地理解整体模式，并允许发现预测异常值。每一行代表一个特征，横坐标为SHAP值。一个点代表一个样本，颜色表示特征值(红色高，蓝色低)。比如，这张图表明LSTAT特征较高的取值会降低预测的房价

```
# summarize the effects of all the features
shap.summary_plot(shap_values, X)
```

#### 5.5 Feature Importance

取每个特征的SHAP值的绝对值的平均值作为该特征的重要性，得到一个标准的条形图(multi-class则生成堆叠的条形图)

```
shap.summary_plot(shap_values, X, plot_type="bar")
```

#### 5.6 Interaction Values

interaction value是将SHAP值推广到更高阶交互的一种方法。树模型实现了快速、精确的两两交互计算，这将为每个预测返回一个矩阵，其中主要影响在对角线上，交互影响在对角线外。这些数值往往揭示了有趣的隐藏关系(交互作用)

```python
shap_interaction_values = explainer.shap_interaction_values(X)
shap.summary_plot(shap_interaction_values, X)
```

#### 5.7 dependence_plot

为了理解单个feature如何影响模型的输出，我们可以将该feature的SHAP值与数据集中所有样本的feature值进行比较。由于SHAP值表示一个feature对模型输出中的变动量的贡献，下面的图表示随着特征RM变化的预测房价(output)的变化。单一RM(特征)值垂直方向上的色散表示与其他特征的相互作用，为了帮助揭示这些交互作用，“dependence_plot函数”自动选择另一个用于着色的feature。在这个案例中，RAD特征着色强调了RM(每栋房屋的平均房间数)对RAD值较高地区的房价影响较小。

```
# create a SHAP dependence plot to show the effect of a single feature across the whole dataset
shap.dependence_plot("RM", shap_values, X)
```

#### 5.8 其他类型的explainers

SHAP库可用的explainers有：

- deep：用于计算深度学习模型，基于DeepLIFT算法
- gradient：用于深度学习模型，综合了SHAP、集成梯度、和SmoothGrad等思想，形成单一期望值方程
- kernel：模型无关，适用于任何模型
- linear：适用于特征独立不相关的线性模型
- tree：适用于树模型和基于树模型的集成算法
- sampling：基于特征独立性假设，当你想使用的后台数据集很大时，kenel的一个很好的替代方案

---------------------------

SHAP proposed a new fair way to measure contribution which is justified in game theory.

> SHAP提出了一种新的衡量贡献的公平方法，并在博弈论中得到了验证。

Understanding why a model makes a certain prediction can be as crucial as the prediction’s accuracy in many applications. However, the highest accuracy for large modern datasets is often achieved by complex models that even experts struggle to interpret, such as ensemble or deep learning models, creating a tension between accuracy and interpretability.

> 在许多应用中，理解为什么一个模型做出某种预测与预测的准确性同样重要。
> 
> 然而，对于大型现代数据集来说，最高的准确性往往是通过复杂的模型来实现的，即使是专家也很难解释这些模型，比如集成或深度学习模型，这在准确性和可解释性之间制造了一种tension。

Simple models are easy to interpret. They built a simple model which works well only on a local point (We don't need to predict on the all points). Then, use the simple model to interpret how it's trained.

> 简单的模型很容易解释。他们建立了一个简单的模型，只在局部点上运行良好(我们不需要预测所有的点)。然后，使用简单的模型来解释它是如何被训练的。

<img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/lime.png" width="420">

The main contribution of SHAP is that they introduced the concept of Shapley Value to measure the contribution.

> SHAP的主要贡献是他们引入了Shapley值的概念来衡量贡献。


- Question: What is a "fair" way for a colition to divide its payoff?
> 什么是一种“公平”的方式来分配它的收益?
  - Depends on the definition of "fairness"
  
> 取决于“公平”的定义

- Approach: Identify axioms that express properties of a fair payoff division
> 方法:确定表示公平回报分配的性质的公理
  - Symmetry: Interchangeable agents should receive the same payments
  
> 对称:可互换的 agents 应收到相同的回报
  
  - Dummy Players: Dummy players should receive nothing
 
> dummy玩家 : dummy玩家应该什么都得不到
  
  - Additivity: $$(v_1 + v_2)(S) = v_1(S) + v_2(S)$$

The author of SHAP found that we can apply this concept to machine learning. In machine learning, *player* is *feature*, and *contribution* is *score*.
> SHAP的作者发现我们可以将这个概念应用到机器学习中。在机器学习中，*player*是*feature*， *contribution*是*score*。

For example, we have 3 features L, M, and N. The shapley values are calculated like below.
> 例如，我们有3个特征L、M和n。shapley值的计算如下所示。

<img src="https://cdn-images-1.medium.com/max/1600/1*DLL5sCQKeVXboAYIvdgwUw.png" width="640">
<img src="https://cdn-images-1.medium.com/max/1600/1*uGjQRe9U0ebC5HxYXAzg3A.png" width="420">

It's the average of combinations of features. Intuitively, it sounds fair.
> 它是特征组合的平均值。直觉上，这听起来很公平。

代码:

```
import shap
shap.initjs()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(val[cols_to_fit])

# SHAP values represent the fair score of features depending on their contribution towards the total score in the set of features.
shap.summary_plot(shap_values, val[cols_to_fit], plot_type='bar')

# SHAP also can visualize how the score changes when the feature value is low/high on each data.
shap.summary_plot(shap_values, val[cols_to_fit], feature_names=cols_to_fit)

``` 

# 特征工程中的聚合操作

### size

```
agg = df.groupby(['groupId']).size().to_frame('players_in_team')
```

### min,max,sum,median,mean,rank

```
def min_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId','groupId'])[features].min()
    return df.merge(agg, suffixes=['', '_min'], how='left', on=['matchId', 'groupId'])

def max_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].max()
    return df.merge(agg, suffixes=['', '_max'], how='left', on=['matchId', 'groupId'])

def sum_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].sum()
    return df.merge(agg, suffixes=['', '_sum'], how='left', on=['matchId', 'groupId'])

def median_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])

def mean_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    return df.merge(agg, suffixes=['', '_mean'], how='left', on=['matchId', 'groupId'])

def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])
```
