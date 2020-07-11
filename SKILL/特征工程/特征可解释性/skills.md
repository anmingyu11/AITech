# 树模型可以输出特征重要性

# 特征重要性的简单方法

- Correlation
- Score gain on a simple model
- Feature importances of Tree models
- Permutation importance
- SHAP values
- Score gain on a complex model

A. Correlation
B. 在一个简单的模型上获得分数
C. 树模型的特征重要性
D. 排列重要性
E. SHAP 值
F. 在复杂模型上获得分数



# Permutation importance

Permutation importance

The basic idea is that observing how much the score decreases when a feature is not available; the method is known as “permutation importance” or “Mean Decrease Accuracy (MDA)”.

> 其基本思想是观察当某个特性不可用时，得分下降了多少;该方法被称为“排列重要性”或“平均降低准确性(MDA)”。


# eli5
```
import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(model, random_state=42).fit(val[cols_to_fit], val[target])
eli5.show_weights(perm, feature_names=list(cols_to_fit))
```

The figure shows the importance of each feature. ELI5 shuffles the target feature instead of removing it to make it useless so that we don't need to re-train the model again. That's why it's represented like `± 0.0033` (standard deviation).
> 图中显示了每个feature的重要性。ELI5对目标特性进行了shuffles，而不是删除它make it useless，这样我们就不需要重新训练模型了。这就是为什么它表示为±0.0033(标准差)

Removing a feature and see the difference... this is what I was doing above, but more reliable! However, there is room to discuss how to define/measure contribution.
> 删除一个feature，看看有什么不同…这就是我在上面所做的，但是更可靠!但是，还有讨论如何定义/度量贡献的空间。


## 5. SHAP values

SHAP proposed a new fair way to measure contribution which is justified in game theory.
> SHAP提出了一种新的衡量贡献的公平方法，并在博弈论中得到了验证。

[A Unified Approach to Interpreting Model
Predictions](https://arxiv.org/pdf/1705.07874.pdf) (2017)

Understanding why a model makes a certain prediction can be as crucial as the prediction’s accuracy in many applications. However, the highest accuracy for large modern datasets is often achieved by complex models that even experts struggle to interpret, such as ensemble or deep learning models, creating a tension between accuracy and interpretability.

> 在许多应用中，理解为什么一个模型做出某种预测与预测的准确性同样重要。
> 然而，对于大型现代数据集来说，最高的准确性往往是通过复杂的模型来实现的，即使是专家也很难解释这些模型，比如集成或深度学习模型，这在准确性和可解释性之间制造了一种tension。

Simple models are easy to interpret. They built a simple model which works well only on a local point (We don't need to predict on the all points). Then, use the simple model to interpret how it's trained.
> 简单的模型很容易解释。他们建立了一个简单的模型，只在局部点上运行良好(我们不需要预测所有的点)。然后，使用简单的模型来解释它是如何被训练的。

<img src="https://raw.githubusercontent.com/marcotcr/lime/master/doc/images/lime.png" width="420">

This method was proposed in this paper: [“Why Should I Trust You?”
Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf) (2016, known as LIME)

The main contribution of SHAP is that they introduced the concept of Shapley Value to measure the contribution.
> SHAP的主要贡献是他们引入了Shapley值的概念来衡量贡献。

Shapley Value is a solution concept in cooperative game theory, proposed in 1953 by [Lloyd Shapley](https://en.wikipedia.org/wiki/Lloyd_Shapley).
> Shapley值是合作博弈论中的一个解概念，由1953年提出

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
> It's the average of combinations of features. Intuitively, it sounds fair.

The implementation is available on GitHub: [slundberg/shap: A unified approach to explain the output of any machine learning model](https://github.com/slundberg/shap)