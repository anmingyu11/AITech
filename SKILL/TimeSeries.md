# [PFS] 日期转换

```
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
# 年的转换
train['month'] = train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))
train['year'] = train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%Y'))
```

# [PFS]相对日期加减法（month）

```
from dateutil.relativedelta import relativedelta

def convert(date_block):
    date = datetime(2013, 1, 1)
    date += relativedelta(months = date_block)
    return (date.month, date.year)
```

# [PFS] 移动平均

```
plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12,center=False).mean(),label='Rolling Mean');
plt.plot(ts.rolling(window=12,center=False).std(),label='Rolling sd');
plt.legend();
```

# [PFS] 数据的Seasonal,Trend,Residual,分解

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

 
#### 乘法模型

```
import statsmodels.api as sm
import matplotlib as mpl

mpl.rcParams["figure.figsize"] = [12, 8]
# multiplicative
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")
#plt.figure(figsize=(16,12))
fig = res.plot()
```	

#### 加法模型

```
# Additive model
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")
#plt.figure(figsize=(16,12))
fig = res.plot()
#fig.show()
```

# Stationary

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

# 去季节性

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

# AR,MA and ARMA models.

AR, MA and ARMA models:
TL: DR version of the models:

MA - Next value in the series is a function of the average of the previous n number of values
AR - The errors(difference in mean) of the next value is a function of the errors in the previous n number of values
ARMA - a mixture of both.
Now, How do we find out, if our time-series in AR process or MA process?

Let's find out!

TL: DR版模型:

- MA -数列中的下一个值是前n个值的平均值的函数
- AR -下一个值的误差(平均值之差)是前n个值的误差的函数
- ARMA -两者的混合。
现在，我们怎么知道，我们的时间序列是AR过程还是MA过程?

# [PFS] 创建新特征，本月销售量与上 i 个月销售量的差

```
for i in range(1,6):
    sales_33month["T_" + str(i)] = sales_33month.item_cnt_month.shift(i)
sales_33month.fillna(0.0, inplace=True)
sales_33month
```