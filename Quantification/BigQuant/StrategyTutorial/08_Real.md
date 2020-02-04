> [https://bigquant.com/community/t/topic/131764](https://bigquant.com/community/t/topic/131764)
> 
> 导语：开发好一个策略且回测收益、风险都达到目标，下一步该做什么呢？本文将详细介绍怎么将开发好的策略通过模拟交易推送每日交易信号。

## 一、提交模拟实盘

#### 第一步：开发出好策略后，在开发界面右上角点击 开始交易。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/7/5/75c7263fcc2f95adf780921e2229d9d173002283.png)

重要的事说三遍，在点击开始交易前请检查：

1. 由于模拟交易需要实现每天更新预测集数据来预测最新日的结果，因此测试集的证券代码列表的开始日期和结束日期请务必确认已经勾选了绑定实盘参数 ！
2. 由于我们训练好的模型通常不希望每天变化，因此训练集的证券代码列表的开始日期和结束日期请务必确认已经取消了绑定实盘参数 ！
3. 由于模拟交易需要实现每天更新预测集数据来预测最新日的结果，如果您的因子计算需要历史数据例如过去5日收盘价的平均值，那么除了当日的收盘价因子数据外必然需要历史前4个交易日的收盘价因子，因此请检查基础特征抽取模块的参数向前取数据天数这个参数是否设置的足够长(例如取10),否则是无法计算当天的过去5日收盘价这个因子值的，会导致模拟交易报错！

![](https://cdn.bigquant.com/community/uploads/default/original/3X/6/6/66debf4ffb0eaadcd06e9a6563f74bb0fd44d599.png)

#### 第二步：选择 模拟交易，点击 下一步。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/c/0/c0a733e35f3030fbd24a2ca914f80f1e228eb296.png)

#### 第三步：输入策略名称，选择实盘类型（暂只支持股票），点击 下一步。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/c/1/c101b589455a9c7ec33f15feda45b76069e4270a.png)

#### 第四步：至此提交模拟交易成功。扫描 BigQuant公众号 ，绑定微信， 实时接收调仓信号（强烈建议绑定）。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/2/2/22448bf89d9f3e9771d4a9996d34be7fb3c81ceb.png)

## 二、查看模拟交易详情

#### 第一步：在导航栏点击 我的交易 查看提交的所有模拟交易详情。

![](https://cdn.bigquant.com/community/uploads/default/optimized/3X/d/f/df9d95d119927a09c679780b3ac12e85bec4a2aa_1_1380x598.png)

- 类型：模拟交易类型，暂只支持股票。
- 状态：表明该策略是否处于模拟交易运行中，可通过操作中的 开始/暂停 按钮控制。
- 累计收益：从开始模拟至今的所有收益。
- 今日收益：最近一个交易日的当日收益。
- 开始时间：策略是什么时候开始进行模拟交易的
- 调仓通知-微信：表明是否开启微信订阅功能，开启之后，每日在微信上将收到策略调仓信号。可点击开启、关闭
- 调仓通知-邮件：表明是否开启邮件订阅功能，开启之后，每日在邮件里将受到策略调仓信号。可点击开启、关闭。
- 分享：表明是否将策略分享至 策略天梯 供其他人查看、付费订阅。
- 操作：有三个按钮，为 开始/暂停、删除 。开始表明开始或重新运行该策略，暂停表明暂停运行该策略，删除是将策略从模拟交易中永久删除（一旦删除不可恢复，请谨慎操作）。

#### 第二步：点击某个 策略名称 查看该模拟交易详情，包括：

整体指标、收益走势、计划交易（即调仓信号）、持仓详情、交易详情、卖出详情、风险指标、策略日志等。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/8/9/89c241794dc61d0dd7fae9c17abdf06fe535ae5e.png)

## 三、查看每日调仓信号

#### 方式一：在 BigQuant 4 访问 我的交易 15，在某交易详情页中查看 计划交易 信息。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/1/4/1489e87b0fd2fe7887b43f6f0f67bb1e381ddb38.png)

#### 方式二：打开模拟交易的 微信调仓信号， 邮箱调仓信号 ，通过微信、邮件消息查看。

![](https://cdn.bigquant.com/community/uploads/default/optimized/3X/f/7/f7bd6d9ab906d8a89f011f4a85da427f048be142_1_562x998.png)

![](https://cdn.bigquant.com/community/uploads/default/original/3X/9/5/956b0cbe97c81446c5e9585bbfbb999ac49a50a9.png)

#### 方式三：通过 模拟交易API 59 获取交易信号并与实盘对接

使用方式请查看通过API获取自己/订阅的模拟交易持仓数据 55。

## 四、BigQuant模拟交易运行时间

对于日线策略，每个交易日收盘后，BigQuant会立即更新数据，再利用最新数据运行所有用户的模拟交易，一般会从17:00持续到23:00。每个用户的模拟交易运行成功、失败均会推送对应的调仓信号、成功失败消息。用户可通过 模拟交易详情 15、 微信调仓信号 、 邮件调仓信号、 模拟交易API 59 等多种方式获取每日调仓信号及相关收益、风险信息。为了消息的及时接收，小编强烈建议您 绑定微信、邮箱 2 。
