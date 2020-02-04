## 导语 :
 
相信进入到“小学”阶段学习的用户，通过学习“学前”阶段的三篇文章，已经对BigQuant的策略开发平台、AI量化策略和我们的平台结构有了较直观的大致了解。
从本文开始，我们将按照AI策略开发的完整流程（共七步），逐步引导大家自己上手在BigQuant平台上快速构建AI策略。

本文首先介绍如何使用证券代码模块指定股票范围和数据起止日期。

**重要的事情说三遍：模块的输入端口有提示需要连线的上游数据类型，两个模块之间的接口不能随意连接，否则会报错！**

![](https://cdn.bigquant.com/community/uploads/default/original/3X/d/6/d663ab9572052517e00a68c31bb45b319e1ae8d3.jpeg)

### 第一步：新建空白可视化AI策略。
### 第二步：添加模块：在模块列表的 数据输入输出 下找到 证券代码列表 模块并拖入画布。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/8/0/8012309985bbdf5f8d55f3bc3b237e6f02e1ec1a.png)


### 第三步：模块属性设置：选中模块，在右侧属性栏中可修改参数。

- 开始时间：训练集的开始时间设置，格式“年-月-日”。
- 结束时间：训练集的结束时间设置，格式“年-月-日”。
- 交易市场：目前支持种类有
	- CN_STOCK_A – A股
	- CN_FUND – 场内基金
	- CN_FUTURE – 期货
	- US_STOCK – 美股
	- HK_STOCK – 港股
如图所示，我们设置训练集数据时间范围是2013-01-01日至2016-12-31日，股票范围为A股所有股票

![](https://cdn.bigquant.com/community/uploads/default/original/3X/e/0/e0cc89cdd2fc7f053b271a0c726e2efcf53ccbd7.png)

测试集的模块设置与训练集类似，只需要将 “开始时间” 和 “结束时间” 设置为“2017-01-01”和“2018-12-31”即可。

![](https://cdn.bigquant.com/community/uploads/default/original/3X/b/5/b5ec3cb74b7c45ca6c055f1457027552229cbbb0.png)

如果我们想指定一个股票池训练或预测，那么只要在股票代码列表中加入相应的股票代码即可，如下图所示：

![](https://cdn.bigquant.com/community/uploads/default/original/3X/2/9/29c600ccfead7f386d22de741a5b0c8ebf76f27a.png)

小结：至此，我们完成了训练集和预测集数据的起止时间和股票范围设置，接下来会进行目标确定、数据标注部分。如果你想进行更复杂的股票筛选条件设置，请查阅专题教程：


