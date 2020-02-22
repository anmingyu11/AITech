>[https://zhuanlan.zhihu.com/p/35423404](https://zhuanlan.zhihu.com/p/35423404)

# 概述

信息熵是信息论和机器学习中非常重要的概念，应用及其广泛，各种熵之间都存在某些直接或间接的联系，本文试图从宏观角度将各种熵穿插起来，方便理解。

本文首先讲解机器学习算法中常用的各种熵的概念、公式、推导，并且联系然后从机器学习算法进行说明熵的应用，最后是简单总结具体算法角度阐述熵的具体应用场景，并配置相应的视频讲解。

希望通过本文能够全面的梳理熵的各方面知识，由于本人水平有限，如写的不好地方，敬请原谅！

# 机器学习常用熵定义

- 熵是什么？
- 熵存在的意义是啥？
- 为什么叫熵？

这是3个非常现实的问题。

答案非常明确：

- 在机器学习中熵是表征**随机变量分布的混乱程度**，分布越混乱，则熵越大，在物理学上表征物质状态的参量之一，也是体系混乱程度的度量；
- 熵存在的意义是度量信息量的多少，人们常常说信息很多，或者信息较少，但却很难说清楚信息到底有多少，这时熵的意义就体现出来了；(注:在信息论的角度来讲)
- 熵词的由来是1923年胡刚复教授根据热温商之意翻译而来，此次不深究。

上面的回答还是显得非常生硬，其实咱们从生活的角度就非常容易理解了。

整个宇宙发展就是一个熵增的过程，具体到细节就是气体扩散、热量传递、宇宙爆炸等等，如果不加干扰，几种气体肯定会混合到一起，任何比环境温度高的物体，都会把热量向低温环境散发，直到系统内温度平衡，所有的恒星终将熄灭，宇宙中不再有能量的流动，因而不可避免地走向无序。

如果房间不去打扫，那么房间肯定越来越乱，这就是一个自然的熵增过程。如果不施加外力影响，事物永远向着更混乱的状态发展，

故而人存在的意义就是通过自身努力改造自身、改造自然。

借用一句话：过去五千年，人类文明的进步只是因为人类学会利用外部能量（牲畜、火种、水力等等），越来越多的能量注入，使得人类社会向着文明有序的方向发展即通过人类的努力使得熵值一直在下降。大家一起努力使得整个世界熵值下降的更多吧！！！

# 1) 自信息

自信息是熵的基础，理解它对后续理解各种熵非常有用。

自信息表示某一事件发生时所带来的信息量的多少，当事件发生的概率越大，则自信息越小，或者可以这样理解：

- 某一事件发生的概率非常小，但是实际上却发生了(观察结果)，则此时的自信息非常大；
- 某一事件发生的概率非常大，并且实际上也发生了，则此时的自信息较小。

以全班的考试成绩为例，通常我们知道整个班成绩是符合高斯分布的，通过一次考试，发现每个人成绩都是相同的，则在学校看来这是一个爆炸性新闻，因为这是一个极低的事件，但是却发生了，不符合常规，下一步应该就是调查了吧。

> 注： 曾经看过一个例子，巴西进世界杯决赛和中国进世界杯决赛的信息量谁更大，私以为觉得这个例子更加简单粗暴一些。

再说一个生活中的例子，如果有人告诉我们一件相当不可能发生的事件发生了，那么我们收到的信息量要多于我们被告知某个很可能发生的事件发生时收到的信息，此时自信息就比较大了。

从通俗角度理解了自信息的含义和作用，但是如何度量它呢？

我们现在要寻找一个函数，它要满足的条件是：

- 事件发生的概率越大，则自信息越小；
- 自信息不能是负值，最小是$0$；
- 自信息应该满足可加性，并且两个独立事件的自信息应该等于两个事件单独的自信息。

下面给出自信息的具体公式：

![自信息公式](/Users/helloword/Anmingyu/KaggleCom/Algorithms/Entropy/eq_self_info.svg)

#### 自信息的图像

![自信息的图](/Users/helloword/Anmingyu/KaggleCom/Algorithms/Entropy/self_info_graph.jpg)

其中**$p_i$表示随机变量的第$i$个事件发生的概率**，自信息单位是$bit$,表征描述该信息需要多少位。可以看出，自信息的计算和随机变量本身数值没有关系，只和其概率有关，同时可以很容易发现上述定义满足自信息的3个条件。

# 2) 信息熵

上述自信息描述的是随机变量的某个事件发生所带来的信息量，**而信息熵通常用来描述整个随机分布所带来的信息量平均值**，更具统计特性。

信息熵也叫香农熵，在机器学习中，由于熵的计算是依据样本数据而来，故也叫经验熵。其公式定义如下：

![](/Users/helloword/Anmingyu/KaggleCom/Algorithms/Entropy/eq_info_ent.svg)
![](/Users/helloword/Anmingyu/KaggleCom/Algorithms/Entropy/eq_info_ent2.svg)

**从公式可以看出，信息熵$H(X)$是各项自信息的累加值，由于每一项都是整正数，故而随机变量取值个数越多，状态数也就越多，累加次数就越多，信息熵就越大，混乱程度就越大，纯度越小。**

越宽广的分布，熵就越大，在同样的定义域内，由于分布 宽广性中脉冲分布<高斯分布<均匀分布，故而熵的关系为 脉冲分布信息熵<高斯分布信息熵<均匀分布信息熵。

可以通过数学证明，当随机变量分布为均匀分布时即状态数最多时，熵最大。

**熵代表了随机分布的混乱程度，这一特性是所有基于熵的机器学习算法的核心思想。**

推广到多维随机变量的联合分布，其联合信息熵为：

![](/Users/helloword/Anmingyu/KaggleCom/Algorithms/Entropy/eq_joint_info_ent.svg)

注意事项：

1. 熵只依赖于随机变量的分布,与随机变量取值无关；
2. 定义$0log0=0$(因为可能出现某个取值概率为$0$的情况)；
3. 熵越大,随机变量的不确定性就越大,分布越混乱，随机变量状态数越多。

# 3) 条件熵

条件熵的定义为：在$X$给定条件下，$Y$的条件概率分布的熵对$X$的数学期望。有点抽象，看具体公式就比较容易理解了：

![](https://www.zhihu.com/equation?tex=%5C%5CH%28Y%7CX%29%3DE_%7Bx%5Csim+p%7D%5BH%28Y%7CX%3Dx%29%5D%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dp%28x%29H%28Y%7CX%3Dx%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dp%28x%29%5Csum_%7Bj%3D1%7D%5E%7Bm%7Dp%28y%7Cx%29log%5Cspace+p%28y%7Cx%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bm%7Dp%28x%2Cy%29log%5Cspace+p%28y%7Cx%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3D-%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bm%7Dp%28x%2Cy%29log%5Cspace+p%28x%2Cy%29-%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bm%7Dp%28x%2Cy%29log%5Cspace+p%28x%29%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3D-%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bm%7Dp%28x%2Cy%29log%5Cspace+p%28x%2Cy%29-%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dlog%5Cspace+p%28x%29%5Csum_%7Bj%3D1%7D%5E%7Bm%7Dp%28x%2Cy%29%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3D-%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Csum_%7Bj%3D1%7D%5E%7Bm%7Dp%28x%2Cy%29log%5Cspace+p%28x%2Cy%29-%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dp%28x%29log%5Cspace+p%28x%29%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3DH%28X%2CY%29-H%28X%29)

同理可得：
![](https://www.zhihu.com/equation?tex=H%28X%7CY%29%3DH%28X%2CY%29-H%28Y%29)

# 4) 交叉熵

对机器学习算法比较熟悉的同学，对交叉熵应该是最熟悉的，其广泛用在逻辑回归的 sigmoid 和 softmax 函数中作为损失函数使用。

其主要用于度量两个概率分布间的差异性信息，由于其和相对熵非常相似，故详细分析对比见下一小结。$p$对$q$的交叉熵表示$q$分布的自信息对$p$分布的期望，公式定义为：

![](https://www.zhihu.com/equation?tex=%5C%5CH%28p%2Cq%29%3DE_%7Bx%5Csim+p%7D%5B-log%5Cspace+q%28x%29%5D%3D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dp%28x%29log%5Cspace+q%28x%29)

其中。$p$是真实样本分布，$q$是预测得到样本分布。

在信息论中，其计算的数值表示：如果用错误的编码方式$q$去编码真实分布$p$的事件，需要多少$bit$数，是一种非常有用的衡量概率分布相似性的数学工具。

由于交叉熵在逻辑回归中应用广泛，这里给出其定义式，使读者知道交叉熵的具体应用。逻辑回归算法的损失函数就是交叉熵，也叫做负对数似然，其定义为：

![](https://www.zhihu.com/equation?tex=%5C%5CJ%28%CE%B8%29%3D-%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28y_ilog%5Cspace+h_%CE%B8%28x_i%29%2B%281-y_i%29log%5Cspace+%281-h_%CE%B8%28x_i%29%29%29)

其中，$y_i$是第$i$个样本的真实标签，$h$是 sigmoid 预测输出值，$J$是凸函数，可以得到全局最优解。

对于多分类的逻辑回归算法，通常我们使用 softmax 作为输出层映射，其对应的损失函数也叫交叉熵，只不过写法有点区别，具体如下：

![](https://www.zhihu.com/equation?tex=%5C%5CJ%28%CE%B8%29%3D-%5Cfrac%7B1%7D%7Bm%7D%5B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Csum_%7Bj%3D1%7D%5E%7Bk%7D1%5Clbrace+y%5E%7Bi%7D%3Dj%5Crbrace+log%5Cspace+%5Cfrac%7Be%5E%7B%CE%B8%5ET_j+x%5Ei%7D%7D%7B%5Csum_%7Bl%3D1%7D%5E%7Bk%7De%5E%7B%CE%B8%5ET_lx%5Ei%7D%7D%5D)

其中，$m$是样本个数,$k$是输出层个数。

![](https://www.zhihu.com/equation?tex=%5C%5CJ%28%CE%B8%29%3D-%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28y_ilog%5Cspace+h_%CE%B8%28x_i%29%2B%281-y_i%29log%5Cspace+%281-h_%CE%B8%28x_i%29%29%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3D-%5Cfrac%7B1%7D%7Bm%7D%5B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Csum_%7Bj%3D0%7D%5E%7B1%7D1%5Clbrace+y%5E%7Bi%7D%3Dj%5Crbrace+log%5Cspace+p%28y%5Ei%3Dj%7Cx%5Ei%3B%CE%B8%29)

可以看出，其实两者是一样的， softmax 只是对 sigmoid 在多分类上面的推广。

# 5）相对熵

相对熵是一个较高端的存在，其作用和交叉熵差不多。相对熵经常也叫做KL散度，在贝叶斯推理中， ![](https://www.zhihu.com/equation?tex=D_%7BKL%7D%28p%7C%7Cq%29) 衡量当你修改了从先验分布$q$到后验分布$p$的信念之后带来的信息增益。首先给出其公式：

![](https://www.zhihu.com/equation?tex=%5C%5CD_%7BKL%7D%28p%7C%7Cq%29%3DE_%7Bx%5Csim+p%7D%5Blog%5Cspace+%5Cfrac+%7Bp%28x%29%7D%7Bq%28x%29%7D%5D%3D-E_%7Bx%5Csim+p%7D%5Blog%5Cspace+%5Cfrac+%7Bq%28x%29%7D%7Bp%28x%29%7D%5D)
![](https://www.zhihu.com/equation?tex=%5C%5C%3D-%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dp%28x%29log%5Cspace+%5Cfrac+%7Bq%28x%29%7D%7Bp%28x%29%7D%3DH%28p%2Cq%29-H%28p%29)

相对熵较交叉熵有更多的优异性质，主要为：

1. 当$p$分布和$q$分布相等时候，KL散度值为0，这是一个非常好的性质
2. 可以证明是非负的
3. 非对称的，通过公式可以看出，KL散度是衡量两个分布的不相似性，不相似性越大，则值越大，当完全相同时，取值为0

**简单对比交叉熵和相对熵，可以发现仅仅差了一个$H(p)$，如果从优化角度来看，$p$是真实分布，是固定值，最小化KL散度情况下，$H(p)$可以省略，此时交叉熵等价于KL散度。**

## 下面讨论一个比较现实且非常重要的问题：既然相对熵和交叉熵表示的含义一样，为啥需要两个？

在机器学习中何时使用相对熵，何时使用交叉熵？要彻底说清这个问题，难度很大，这里我仅仅从我知道的方面讲讲。

首先需要明确：**在最优化问题中，最小化相对熵等价于最小化交叉熵；相对熵和交叉熵的定义其实都可以从最大似然估计得到**

下面进行详细推导：以某个生成模型算法为例，假设是生成对抗网络GAN，其实只要是生成模型，都满足以下推导。

若给定一个样本数据的真实分布 ![](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 和生成的数据分布 ![](https://www.zhihu.com/equation?tex=P_G%28x%3B%CE%B8%29) ，那么生成模型希望能找到一组参数$θ$使分布 ![](https://www.zhihu.com/equation?tex=P_G%28x%3B%CE%B8%29) 和 ![](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 之间的距离最短，也就是找到一组生成器参数而使得生成器能生成十分逼真的分布。

现在从真实分布 ![](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 中抽取$m$个真实样本 ![](https://www.zhihu.com/equation?tex=%7Bx%5E1%2Cx%5E2%2C...x%5Em%7D) ,对于每一个真实样本，我们可以计算 ![](https://www.zhihu.com/equation?tex=P_G%28x%5Ei%3B%CE%B8%29) ，即在由$θ$确定的生成分布中， ![](https://www.zhihu.com/equation?tex=x%5Ei) 样本所出现的概率。因此，我们可以构建似然函数：

![](https://www.zhihu.com/equation?tex=%5C%5CL%3D%5Cprod_%7Bi%3D1%7D%5E%7Bm%7DP_G%28x%5Ei%3B%CE%B8%29)

最大化似然函数，即可求得最优参数 ![](https://www.zhihu.com/equation?tex=%CE%B8%5E%2A) :

![](https://www.zhihu.com/equation?tex=%5C%5C%CE%B8%5E%2A%3Darg%5Cspace+%5Cunderbrace%7Bmax%7D_%7B%CE%B8%7D%5Cprod_%7Bi%3D1%7D%5E%7Bm%7DP_G%28x%5Ei%3B%CE%B8%29)

转换为对数似然函数：

![](https://www.zhihu.com/equation?tex=%5C%5C%CE%B8%5E%2A%3Darg%5Cspace+%5Cunderbrace+%7Bmax%7D_%7B%CE%B8%7D%5Cspace+log%5Cprod_%7Bi%3D1%7D%5E%7Bm%7DP_G%28x%5Ei%3B%CE%B8%29)

![](https://www.zhihu.com/equation?tex=%5C%5C%3Darg%5Cspace+%5Cunderbrace+%7Bmax%7D_%7B%CE%B8%7D%5Cspace+%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dlog%5Cspace+P_G%28x%5Ei%3B%CE%B8%29)

由于是求最大值，故整体乘上常数对结果没有影响,这里是逐点乘上一个常数，所以不能取等于号，但是因为在取得最大值时候 ![](https://www.zhihu.com/equation?tex=P_G%28x%3B%CE%B8%5E%2A%29) 和 ![](https://www.zhihu.com/equation?tex=P_%7Bdata%7D%28x%29) 肯定是相似的，并且肯定大于$0$，所以依然可以认为是近似相等的

![](https://www.zhihu.com/equation?tex=%5C%5C%5Capprox+arg%5Cspace+%5Cunderbrace+%7Bmax%7D_%7B%CE%B8%7D%5Cspace+%5Csum_%7Bi%3D1%7D%5E%7Bm%7DP_%7Bdata%7D%28x%5Ei%29log%5Cspace+P_G%28x%5Ei%3B%CE%B8%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3Darg%5Cspace+%5Cunderbrace+%7Bmax%7D_%7B%CE%B8%7D%5Cspace+E_%7Bx%5Csim%7Bp_%7Bdata%7D%7D%7D%5Blog%5Cspace+P_G%28x%5Ei%3B%CE%B8%29%5D)

上面的公式正好是交叉熵的定义式。然后我们再该基础上减掉一个常数，

![](https://www.zhihu.com/equation?tex=%5C%5C%CE%B8%5E%2A%3Darg%5Cspace+%5Cunderbrace+%7Bmax%7D_%7B%CE%B8%7D%5Cspace+%5Clbrace+E_%7Bx%5Csim%7Bp_%7Bdata%7D%7D%7D%5Blog%5Cspace+P_G%28x%5Ei%3B%CE%B8%29%5D-E_%7Bx%5Csim%7Bp_%7Bdata%7D%7D%7D%5Blog%5Cspace+P_%7Bdata%7D%28x%5Ei%29%5D)
![](https://www.zhihu.com/equation?tex=%5C%5C%3Darg%5Cspace+%5Cunderbrace+%7Bmax%7D_%7B%CE%B8%7D%5Cspace+%5Cint_%7Bx%7DP_%7Bdata%7D%5Cspace+log+%5Cfrac%7BP_G%28%CE%B8%29%7D%7BP_%7Bdata%7D%7Ddx)
![](https://www.zhihu.com/equation?tex=%5C%5C%3Darg%5Cspace+%5Cunderbrace+%7Bmin%7D_%7B%CE%B8%7D%5Cint_%7Bx%7DP_%7Bdata%7D%5Cspace+log+%5Cfrac%7BP_%7Bdata%7D%7D%7BP_G%28%CE%B8%29%7Ddx%5Cspace+)
![](https://www.zhihu.com/equation?tex=%5C%5C%3Darg%5Cspace+%5Cunderbrace+%7Bmin%7D_%7B%CE%B8%7DE_%7Bx%5Csim+P_%7Bdata%7D%7D%5Blog%5Cspace+%5Cfrac%7BP_%7Bdata%7D%7D%7BP_G%28%CE%B8%29%7D%5D)
![](https://www.zhihu.com/equation?tex=%5C%5C%3Darg%5Cspace+%5Cunderbrace+%7Bmin%7D_%7B%CE%B8%7DKL%28P_%7Bdata%7D%28x%29%7C%7CP_G%28x%3B%CE%B8%29%29)

通过以上各公式可以得出以下结论：**最大化似然函数，等价于最小化负对数似然，等价于最小化交叉熵，等价于最小化KL散度。**

推导了半天，依然没有回答上面的问题。学过机器学习的同学都知道：交叉熵大量应用在 sigmoid 函数和 softmax 函数中，最典型的算法应该就是神经网络和逻辑回归吧，而相对熵大量应用在生成模型中，例如GAN、EM、贝叶斯学习和变分推导中。

从这里我们可以看出一些端倪

- 如果想通过算法对样本数据进行概率分布建模，那么通常都是使用相对熵，因为我们需要明确的知道生成的分布和真实分布的差距，最好的KL散度值应该是$0$；

- 而在判别模型中，仅仅只需要评估损失函数的下降值即可，交叉熵可以满足要求，其计算量比KL散度小。

在数学之美书中，有这样几句话：交叉熵，其用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小，相对熵，其用来衡量两个取值为正的函数或概率分布之间的差异。但是我觉得依然看不出区别。

# 6）互信息

互信息在信息论和机器学习中非常重要，其可以评价两个分布之间的距离，这主要归因于其对称性，假设互信息不具备对称性，那么就不能作为距离度量，例如相对熵，由于不满足对称性，故通常说相对熵是评价分布的相似程度，而不会说距离。

互信息的定义为：**一个随机变量由于已知另一个随机变量而减少的不确定性**，或者说从贝叶斯角度考虑，由于新的观测数据y到来而导致x分布的不确定性下降程度。公式如下：

![](https://www.zhihu.com/equation?tex=%5C%5CI%28X%2CY%29%3DH%28X%29-H%28X%7CY%29%3DH%28Y%29-H%28Y%7CX%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3DH%28X%29%2BH%28Y%29-H%28X%2CY%29)
![](https://www.zhihu.com/equation?tex=%5C%5C%3DH%28X%2CY%29-H%28X%7CY%29-H%28Y%7CX%29)

具体推导由于比较简单，但是非常繁琐，此次省略。

从公式中可以看出互信息是满足对称性的，**其在特性选择、分布的距离评估中应用非常广泛，请务必掌握**。其实互信息和相对熵也存在联系，如果说相对熵不能作为距离度量，是因为其非对称性，那么互信息的出现正好弥补了该缺陷，使得我们可以计算任意两个随机变量之间的距离，或者说两个随机变量分布之间的相关性、独立性。
$$
I(X,Y) = KL(p(x,y)||p(x)p(y))
$$
互信息也是大于等于$0$的，当且仅当$x$与$y$相互独立时候取等号。

# 7）信息增益

信息增益是决策树ID3算法在进行特征切割时使用的划分准则，其物理意义和互信息完全相同，并且公式也是完全相同。其公式如下：

![](https://www.zhihu.com/equation?tex=%5C%5Cg%28D%2CA%29%3DH%28D%29-H%28D%7CA%29)

其中$D$表示数据集，$A$表示特征，信息增益表示得到$A$的信息而使得类$X$的不确定度下降的程度，在ID3中，需要选择一个$A$使得信息增益最大，这样可以使得分类系统进行快速决策。

需要注意的是：在数值上，信息增益和互信息完全相同，但意义不一样，需要区分，当我们说互信息时候，两个随机变量的地位是相同的，可以认为是纯数学工具，不考虑物理意义，当我们说信息增益时候，是把一个变量看成是减少另一个变量不确定度的手段。

# 8) 信息增益率

信息增益率是决策树C4.5算法引入的划分特征准则，其主要是克服信息增益存在的在某种特征上分类特征细，但实际上无意义取值时候导致的决策树划分特征失误的问题。

例如假设有一列特征是身份证ID，每个人的都不一样，其信息增益肯定是最大的，但是对于一个情感分类系统来说，这个特征是没有意义的，此时如果采用ID3算法就会出现失误，而C4.5正好克服了该问题。其公式如下：
$$
g_r(D,A) = g(D,A) / H(A)
$$

# 9) 基尼系数

基尼系数是决策树 CART 算法引入的划分特征准则，**其提出的目的不是为了克服上面算法存在的问题，而主要考虑的是计算快速性、高效性，这种性质使得 CART 二叉树的生成非常高效**。其公式如下：

![](https://www.zhihu.com/equation?tex=%5C%5CGini%28p%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dp_i%281-p_i%29%3D1-%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dp%5E2_i)
![](https://www.zhihu.com/equation?tex=%5C%5C%3D1-%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%28%5Cfrac%7B%7CC_k%7C%7D%7B%7CD%7C%7D%29%5E2)

可以看出，基尼系数越小，表示选择该特征后熵下降最快，对分类模型效果更好，其和信息增益和信息增益率的选择指标是相反的。

基尼系数主要是度量数据划分对训练数据集$D$的不纯度大小，基尼系数越小，表明样本的纯度越高。

这里还存在一个问题，这个公式显得非常突兀，感觉突然就出来了，没有那种从前人算法中改进而来的感觉？其实为啥说基尼系数计算速度快呢，因为基尼系数实际上是信息熵的一阶进似，作用等价于信息熵，只不过是简化版本。

根据泰勒级数公式，将 $ f(x) = - ln(x)$ 在 $x = 1$处展开，忽略高阶无穷小，其可以等价为  $f(x) = 1 - x$ ，所以可以很容易得到上述定义。

### 常用的泰勒公式

![](/Users/helloword/Anmingyu/KaggleCom/Algorithms/Entropy/Taylor1.png)

![](/Users/helloword/Anmingyu/KaggleCom/Algorithms/Entropy/Taylor2.png)

# 总结

- 自信息是衡量随机变量中的某个事件发生时所带来的信息量的多少，越是不可能发生的事情发生了，那么自信息就越大；
- 信息熵是衡量随机变量分布的混乱程度，是随机分布各事件发生的自信息的期望值，随机分布越宽广，则熵越大，越混乱；
- 信息熵推广到多维领域，则可得到联合信息熵；
- 在某些先验条件下，自然引出条件熵，其表示在X给定条件下，$Y$的条件概率分布熵对$X$的数学期望，没有啥特别的含义，是一个非常自然的概念；
- 前面的熵都是针对一个随机变量的，而交叉熵、相对熵和互信息可以衡量两个随机变量之间的关系，三者作用几乎相同，只是应用范围和领域不同。
- 交叉熵一般用在神经网络和逻辑回归中作为损失函数
- 相对熵一般用在生成模型中用于评估生成的分布和真实分布的差距
- 而互信息是纯数学的概念，作为一种评估两个分布之间相似性的数学工具，其三者的关系是：
	- 最大化似然函数，等价于最小化负对数似然，等价于最小化交叉熵，等价于最小化KL散度，互信息相对于相对熵区别就是互信息满足对称性；
- 作为熵的典型机器学习算法-决策树，广泛应用了熵进行特征划分，常用的有信息增益、信息增益率和基尼系数。
