>[https://www.kaggle.com/c/competitive-data-science-predict-future-sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)

# Predict Future Sales

您将获得每日历史销售数据。任务是预测测试集在每个商店中出售的产品总数。请注意，商店和产品的清单每月都会略有变化。创建可以处理此类情况的robust模型是挑战的一部分。

#### File
- `sales_train.csv` - 训练集。2013年1月至2015年10月的每日历史数据。
- `test.csv` - 测试集。您需要预测这些商店和产品在2015年11月的销售额。
- `sample_submission.csv` - 格式正确的示例提交文件。
- `items.csv` - 有关项目/产品的补充信息。
- `item_categories.csv` -   有关项目类别的补充信息。
- `shop.csv` - 有关商店的补充信息。

#### Data
- `ID` -  代表测试集中的（商店，商品）元组的ID
- `shop_id` - 商店的唯一标识符
- `item_id` - 产品的唯一标识符
- `item_category_id` - 项目类别的唯一标识符
- `item_cnt_day` - 销售的产品数量。您正在预测此量度的每月金额
- `item_price` - 商品的当前价格
- `date` - 格式为dd / mm / yyyy的日期
- `date_block_num` - 连续的月份号，为方便起见。2013年1月为0,2013年2月为1，...，2015年10月为33
- `item_name` -  项目名称
- `shop_name` - 商店名称
- `item_category_name` - 项目类别名称


#### 1. 传统的时间序列分析
#### 2. 一个很好的Kernel所给与的提示
#### 3. 混合模型的Stacking
#### 4. 深度学习
#### 5. 本竞赛涉及的特征工程技巧
#### 6. 特征工程

# 1) 时间序列分析

## RollingMean

移动平均的写法

```
plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');
plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');
plt.legend();
```

## Seasonal: Trend,Seasonal,residual

we assume an additive model, then we can write

yt = St + Tt + Et

- yt is the data at period t, 
- St is the seasonal component at period t
- Tt is the trend-cycle component at period tt 
- Et is the remainder (or irregular or error) component at period t Similarly for Multiplicative model,

yt = St x Tt x Et


>- yt是t时段的数据，
- St为t时段的季节分量
- Tt是Tt时段的趋势周期分量
- Et是t时段的剩余(或不规则或误差)分量，

 
### 乘法模型

```
import statsmodels.api as sm
import matplotlib as mpl

mpl.rcParams["figure.figsize"] = [12, 8]
# multiplicative
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")
#plt.figure(figsize=(16,12))
fig = res.plot()
```	

### 加法模型

```
# Additive model
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()
```

## Stationary

平稳性：  

1. Series的平均值不应该是时间的函数。 下面的红色图形不是平稳的，因为平均值随时间增加
2. Series的方差不应该是时间的函数。 这个属性被称为方差齐性。 注意红色图中随时间变化的点差数据
3. 最后，第i项和第（i + m）项的协方差不应该是时间的函数。在下面的图表中，您会注意到随着时间的增加，价差越来越近。 因此，“红色series”的协方差不是随时间变化的

![](https://static1.squarespace.com/static/53ac905ee4b003339a856a1d/t/5818f84aebbd1ac01c275bac/1478031479192/?format=750w)

平稳性是指序列的时间不变性。 

（即）时间序列中的两个点彼此之间的关联仅取决于它们之间的距离，而不取决于方向（向前/向后）当时间序列是稳定的时，可以更容易建模。 

统计建模方法假定或要求时间序列是平稳的。


There are multiple tests that can be used to check stationarity.

- ADF( Augmented Dicky Fuller Test)
- KPSS
- PP (Phillips-Perron test)

我们来执行最常用的ADF。

ADF 测试来检验平稳性

```
# Stationarity tests
def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(ts)
```

## 去季节性

消除季节性

一旦确定了季节性，就可以对其进行建模。

季节性模型可以从时间序列中删除。此过程称为“ 季节性调整 ”或“反季节化”。

去除了季节性成分的时间序列称为季节性平稳。具有明显季节性成分的时间序列称为非平稳时间序列。

在时间序列分析领域中，有许多复杂的方法可以研究时间序列中的季节性并从中提取季节性。由于我们主要对预测建模和时间序列预测感兴趣，因此我们仅限于可以在历史数据上开发并在对新数据进行预测时可用的方法。


```
# to remove trend
from pandas import Series as Series

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob
```

```

ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')

# Original
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)

# DeTrend
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts)
plt.plot(new_ts)
plt.plot()

# De seasonalization
plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts,12)       # assuming the seasonality is 12 months long
plt.plot(new_ts)
plt.plot()
```

现在在转换之后，我们的DF测试的 p-value < 5%。
因此我们可以假设Series是具有平稳性的
我们可以用上面定义的逆变换函数很容易地得到原始的Series。

# 2) 一个很好的Kernel给予的启示

#### 1. boxplot 是分析异常值的好工具
#### 2. 能够降低数据 memory used 就尽量降低
#### 3. 对数据有一个大略的了解

```
print("----------Top-5- Record----------")
display(sale_train.head(5))
print("-----------Information-----------")
display(sale_train.info())
print("-----------Data Types-----------")
display(sale_train.dtypes)
print("----------Missing value-----------")
display(sale_train.isnull().sum())
print("----------Null value-----------")
display(sale_train.isna().sum())
print("----------Shape of Data----------")
display(sale_train.shape)
```  

#### 4. 对数据的探索是获得数据胜利的关键秘诀之一，简单的特征工程，加减乘除算比例交叉等等是无法让你在kaggle竞赛中获取胜利的，如果能够通过自己的生活常识，和对特征的观察，能够压榨出有实际意义的特征，比如泰坦尼克号的的甲板号，预测未来收入的商店所在城市，商品名称，共享单车的注册用户和非注册用户，这些特征对整个模型的预测都有非常大的程度，此处就像过去在玩《模拟城市》，就像星际争霸2里的serral，你对特征处理的越细，能够获得的分数越高。
#### 5. 特征工程1：先验知识与机器学习相结合往往会得到更好的结果（商店的开关门日期），量纲统一（比如，货币种类不一样，如果这里有两个特征列，一个是美元，一个是卢布，那么可以做特征工程来将这两种货币统一化），商品打折会导致销量上涨，但是同样的商品id价格却发生了变化。
#### 6. 特征工程2：对于日期能够提取出来的特征，是否是节假日，每个月的天数，都会影响商品的销量。
#### 7. 特征工程3：对于重复数据的处理需要慎重考虑。`duplicated`
#### 8. 特征工程4：个人的想法，是否对于一些有意义的特征列，特征列的属性是个字符串，即自然语言，是否可以用自然语言处理的技术来提取有意义的特征。
#### 9. Dataleakage：将测试集的数据和训练集的数据结合起来分析，利用训练数据提供的信息是获取kaggle竞赛胜利的重要秘诀,学名叫做dataleakage.

# 6. 特征工程

### 1. 数据清洗

创建 date_month_block × shop_id × item_id
的表

### 2. 对文本特征的分析

1. `shop_name`: 可以获取商店所在的城市名,并将城市名 LabelEncoder
2. `item_category_name`: 可以获取到物品类别的大类和子类, LabelEncoder

### 3. 时间序列特征

#### Lag特征

lag=[1,2,3,6,12] 

比如 date_block=33 那么计算 33 - lag 的 item_cnt 

分别计算了商品销售的lag和商店销售总量的lag

#### Trend特征 ： 

1. 商品的月价格相对于商品月平均价格的波动比率
2. 商店的月收入相对于商店月平均收入的波动比率

上面这些都是进行标准化过的，更合理一些，虽说用tree模型是不需要量纲统一的，但是经验告诉我们，统一的要比不统一的效果好

#### 时间序列上的移动mean,sum,max,min,std



### 4. 平均值特征

1. 每个月的所有商品总销量
2. 商品对应的月平均销量
3. 每个月商店的所有商品总销量
4. 一个商品类的月平均销量
5. 商店对应的商品类别的月平均销量
6. 每个商店对应的typecode的月平均销量
7. 每个商店对应的subtype_code的月平均销量
8. 每个城市的商品的月平均销量
9. 每个商品在每个城市的月平均销量
10. typecode和subtypecode对应的月平均销量
11. 商品的平均价格
12. 每个月对应的商品均价

#### 5. 其他

- 年，月，日
- Unitary item prices 单项价格 item_price / item_cnt

# 3)混合模型的Stacking

![](https://raw.githubusercontent.com/dimitreOliveira/MachineLearning/master/Kaggle/Predict%20Future%20Sales/Ensemble%20Kaggle.jpg)

### Ensembling

To combine the 1st level model predictions, I'll use a simple linear regression.

As I'm only feeding the model with predictions I don't need a complex model.

Ensemble architecture:

#### 1st level:
- Catboost
- XGBM
- Random forest
- Linear Regression
- KNN

#### 2nd level;
- Linear Regression

## 技巧

#### 运行外部命令

```
!ls ../data/*
```

#### 先看要干啥

```
# for kaggle competition, always look at sample_submission.csv first, so you know what you want to get
# then train.csv and test.csv
sub = pd.read_csv('../data/sample_submission.csv')
sub.head()
```