> https://www.kaggle.com/c/bike-sharing-demand/data

# 共享单车的需求预测

## 数据描述

You are provided hourly rental data spanning two years. For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. You must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

Data Fields

- datetime - hourly date + timestamp  
- season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
- holiday - whether the day is considered a holiday
- workingday - whether the day is neither a weekend nor holiday
- weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
- temp - temperature in Celsius
- atemp - "feels like" temperature in Celsius
- humidity - relative humidity
- windspeed - wind speed
- casual - number of non-registered user rentals initiated
- registered - number of registered user rentals initiated
- count - number of total rentals

### 共8点
理论:

1) RMLSE.

2) 切比雪夫不等式.

3) 多重共线性.

工具:

4) missingno.

经验:

5) 偏斜数据处理.

6) 时间序列分析.

7) 如何进行手动调参，以及类似这种数据集的交叉验证细节, StratifiedKFold.

8) 对于无法正确得到预测结果的数据应该怎么办.

9) 加权的模型融合提高准确度.

## 1) RMSLE
均方误差 （mean square error）MSE

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>M</mi>
  <mi>S</mi>
  <mi>E</mi>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>n</mi>
  </mfrac>
  <munderover>
    <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>n</mi>
    </mrow>
  </munderover>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>y</mi>
    <mi>i</mi>
  </msub>
  <mo>&#x2013;</mo>
  <msub>
    <mrow class="MJX-TeXAtom-ORD">
      <mover>
        <mi>y</mi>
        <mo stretchy="false">&#x005E;<!-- ^ --></mo>
      </mover>
    </mrow>
    <mi>i</mi>
  </msub>
  <msup>
    <mo stretchy="false">)</mo>
    <mn>2</mn>
  </msup>
</math>
 

均方根误差 （root mean squared error）RMSE

均方根误差是均方误差开根号得到的结果

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>R</mi>
  <mi>M</mi>
  <mi>S</mi>
  <mi>E</mi>
  <mo>=</mo>
  <msqrt>
    <mfrac>
      <mn>1</mn>
      <mi>n</mi>
    </mfrac>
    <munderover>
      <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>i</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>n</mi>
      </mrow>
    </munderover>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>y</mi>
      <mi>i</mi>
    </msub>
    <mo>&#x2013;</mo>
    <msub>
      <mrow class="MJX-TeXAtom-ORD">
        <mover>
          <mi>y</mi>
          <mo stretchy="false">&#x005E;<!-- ^ --></mo>
        </mover>
      </mrow>
      <mi>i</mi>
    </msub>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
  </msqrt>
</math>
 

平均绝对误差 （mean absolute error） MAE

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>M</mi>
  <mi>A</mi>
  <mi>E</mi>
  <mo>=</mo>
  <mfrac>
    <mn>1</mn>
    <mi>n</mi>
  </mfrac>
  <munderover>
    <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>i</mi>
      <mo>=</mo>
      <mn>1</mn>
    </mrow>
    <mrow class="MJX-TeXAtom-ORD">
      <mi>n</mi>
    </mrow>
  </munderover>
  <mo fence="false" stretchy="false">&#x007C;<!-- | --></mo>
  <msub>
    <mi>y</mi>
    <mi>i</mi>
  </msub>
  <mo>&#x2013;</mo>
  <msub>
    <mrow class="MJX-TeXAtom-ORD">
      <mover>
        <mi>y</mi>
        <mo stretchy="false">&#x005E;<!-- ^ --></mo>
      </mover>
    </mrow>
    <mi>i</mi>
  </msub>
  <mo fence="false" stretchy="false">&#x007C;<!-- | --></mo>
</math>

均方根对数误差 （root mean squared logarithmic error） RMSLE

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>R</mi>
  <mi>M</mi>
  <mi>S</mi>
  <mi>L</mi>
  <mi>E</mi>
  <mo>=</mo>
  <msqrt>
    <mfrac>
      <mn>1</mn>
      <mi>n</mi>
    </mfrac>
    <munderover>
      <mo movablelimits="false">&#x2211;<!-- ∑ --></mo>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>i</mi>
        <mo>=</mo>
        <mn>1</mn>
      </mrow>
      <mrow class="MJX-TeXAtom-ORD">
        <mi>n</mi>
      </mrow>
    </munderover>
    <mo stretchy="false">(</mo>
    <mi>l</mi>
    <mi>o</mi>
    <mi>g</mi>
    <mo stretchy="false">(</mo>
    <msub>
      <mrow class="MJX-TeXAtom-ORD">
        <mover>
          <mi>y</mi>
          <mo stretchy="false">&#x005E;<!-- ^ --></mo>
        </mover>
      </mrow>
      <mi>i</mi>
    </msub>
    <mo>+</mo>
    <mn>1</mn>  f
    <mo stretchy="false">)</mo>
    <mo>&#x2013;</mo>
    <mi>l</mi>
    <mi>o</mi>
    <mi>g</mi>
    <mo stretchy="false">(</mo>
    <msub>
      <mi>y</mi>
      <mi>i</mi>
    </msub>
    <mo>+</mo>
    <mn>1</mn>
    <mo stretchy="false">)</mo>
    <msup>
      <mo stretchy="false">)</mo>
      <mn>2</mn>
    </msup>
  </msqrt>
</math>

使用RMSLE的好处一：

  假如真实值为1000，若果预测值是600，那么RMSE=400， RMSLE=0.510

  假如真实值为1000，若预测结果为1400， 那么RMSE=400， RMSLE=0.336

  可以看出来在均方根误差相同的情况下，预测值比真实值小这种情况的错误比较大，即对于预测值小这种情况惩罚较大。

使用RMSLE的好处二：

  直观的经验是这样的，当数据当中有少量的值和真实值差值较大的时候，使用log函数能够减少这些值对于整体误差的影响。

代码:

```
def rmsle(y, y_,convertExp=True):
    if convertExp:
        y = np.exp(y),
        y_ = np.exp(y_)
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))
```

## 2) 通过切比雪夫不等式来排除异常值

### 概念

这个不等式以数量化这方式来描述，究竟“几乎所有”是多少，“接近”又有多接近：

- 与平均相差2个标准差以上的值，数目不多于1/4
- 与平均相差3个标准差以上的值，数目不多于1/9
- 与平均相差4个标准差以上的值，数目不多于1/16
- ……

与平均相差k个标准差以上的值，数目不多于1/k2

举例说，若一班有36个学生，而在一次考试中，平均分是80分，标准差是10分，我们便可得出结论：少于50分（与平均相差3个标准差以上）的人，数目不多于4个（=36*1/9）。

### 排除异常值的代码
```python
def chebychev_normal(df,k):
    mean = np.mean(df)
    std = np.std(df)
    return (df < (mean + k * std)) & (df > (mean - k * std))
```

```python
dailyDataWithoutOutliers = dailyData[np.abs(dailyData["count"]-dailyData["count"].mean())<=(3*dailyData["count"].std())] 
```

## 3) 多重共线性

### 通过进行连续型数据的双变量分析得到
atemp 和temp具有多重共线性

代码：

```python
cont_names=['temp','atemp','humidity','windspeed']

i=1
for name_1 in cont_names:
    j=cont_names.index(name_1) # col index 

    while(j<len(cont_names)-1):
        ax = plt.subplot(6,1,i) # 6 rows 1 cols 
        ax.set_title(name_1+' vs '+cont_names[j+1])
        sns.jointplot(x=name_1,y=cont_names[j+1],data=X_train) 
        j=j+1
        i=i+1
        plt.show()
```

通过观察散点图可以看出temp和atempyou多重共线性

Not much can be inferred about the distribution of these variables except for variable 'temp' and 'atemp' that almost have

> 除了变量“ temp”和“ atemp”几乎具有相似的上下文外，无法推断出这些变量的分布情况。

similar context. We would be using the 'temp' and getting rid of the 'atemp' variables for better precision value and avoiding multi-collinearity.

> 我们将使用'temp'并摆脱'atemp'变量以获得更好的精度值并避免多重共线性。


#### 多重共线性本身不是问题。通常来说多重共线性的解决办法主要有以下几种:

1. 什么都不做，这是最好的办法~如果多重共线性不影响你核心变量的显著度。
2. 做岭回归，岭回归会有惩罚，估计量有偏，实际应用中貌似用的比较少。
3. 改写模型，做个差分啊，ln啊啥的，把绝对值变成了增长率。
4. 增加样本量。多重共线性最大的问题不就是方差会大啊，因为多重共线性会使方差公式中那个分母Rij方变大，那增加样本量使分母中xi-x拔变大，抵消了方差变大的影响。

> 链接：https://www.zhihu.com/question/55089869/answer/142699821

#### 多重共线性反映在最后一项上，也就是说是的系数的方差变大了。

注意多重共线性并不意味着假设检验的完全失效，实际上，如果原假设为真，我们的假设检验不会错，size永远是对的，或者说犯第一类错误的概率总是能控制的；但是如果我们的原假设为假，多重共线性导致power大大降低，所以很容易犯第二类错误。

翻译成人话就是，多重共线性会使得你更容易得到不显著的结果。另外还有一个推论就是，如果你得到了显著的结果，也就不用去管什么多重共线性的问题了。这也就是为什么我个人感觉审稿人拿多重共线性说事都是耍流氓：拿去审的稿件基本上不会不显著，如果人家的结果显著了还怀疑多重共线性的话，只能说审稿人自己统计没学好。


--------------------------------------------------------

## 4) 缺失值分析
missing no 可以进行缺失值分析

--------------------------------------------------------

## 5) 对于预测目标高偏斜数据，应该如何处理

As this is a highly skewed data, we will try to transform this data using either log, square-root or box-cox transformation.

> 由于这是一个高度偏斜的数据，因此我们将尝试使用对数，平方根或box-cox转换来转换此数据。

After trying out all three, log square gives the best result. Also as the evaluation metric is NLMSE, using log would help as it would allow to less penalize the large difference in final variable values.

> 在尝试了所有三个之后，对数平方给出最佳结果。同样，由于评估指标为NLMSE，因此使用
log会有所帮助，因为它将减少对最终变量值的较大差异的惩罚。

```
# 在这样转换回来
np.exp(pred_registered) - 1
```

```
def translate_log(X):
    for col in ['casual','registered','count']:
        X['%s_log' % col] = np.log(X[col] + 1)
    return X
```

### log之后的效果

极度左偏的数据变得像正态

```
def plot_dist():
    # count
    fig = plt.figure(figsize=(12,12))
    ax = plt.subplot(321)
    sns.distplot(X_train['count'],ax=ax)
    ax.set_xlabel('count before log')
    ax = plt.subplot(322)
    sns.distplot(np.log(X_train['count'] + 1),ax=ax)
    ax.set_xlabel('count after log')
    
    # casual
    ax = plt.subplot(323)
    sns.distplot(X_train['casual'],ax=ax)
    ax.set_xlabel('count before log')
    ax = plt.subplot(324)
    sns.distplot(np.log(X_train['casual'] + 1),ax=ax)
    ax.set_xlabel('count after log')

    # register
    ax = plt.subplot(325)
    sns.distplot(X_train['registered'],ax=ax)
    ax.set_xlabel('count before log')
    ax = plt.subplot(326)
    sns.distplot(np.log(X_train['registered'] + 1),ax=ax)
    ax.set_xlabel('count after log')

plot_dist()
```

## 6) 时间序列分析

### 计算移动平均

#### 移动平均的概念
移动平均（英语：moving average，MA），又称“移动平均线”，简称均线，是技术分析中一种分析时间序列数据的工具。最常见的是利用股价、回报或交易量等变数计算出移动平均。

移动平均可抚平短期波动，反映出长期趋势或周期。数学上，移动平均可视为一种卷积。

移动平均的计算方法主要有：

- 简单移动平均
- 加权移动平均
- 指数移动平均

#### 代码实现计算移动平均
```
from datetime import datetime
import gc

display(X_train['datetime'])
X_train_cop = X_train.copy()
X_train_cop['datetime'] = X_train_cop['datetime'].apply(lambda x:datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S'))
time_series_df= X_train_cop
time_series_df.index=X_train_cop['datetime']
display(time_series_df['datetime'])

import matplotlib.pyplot as plt

#Applying rolling average on a period of 60 days, as the typical weather lasts for around 3 months (20 days in training data of each month)
plt.figure(figsize=(12,12))
plt.plot(time_series_df[['datetime','count']].rolling(60)['count'].mean())
plt.show()

del time_series_df
del X_train_cop
gc.collect()
```

### 拆分时间戳

```python
from datetime import datetime

#converting string dattime to datetime

X_train['datetime']=X_train['datetime'].apply(lambda x:datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S'))

new_df=X_train

new_df['month']=new_df['datetime'].apply(lambda x:x.month)
new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df['day']=new_df['datetime'].apply(lambda x:x.day)
new_df['year']=new_df['datetime'].apply(lambda x:x.year)
#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)
new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))

```

通过拆分时间戳，分别以年月日小时周等作为单位分析与target的相关性
对年月日天气季节小时等进行one-hot,这些代表(yes/no)的关系都可以进行one_hot,而非量级

### 三变量的eda

```
sns.swarmplot(x='hour',y='temp',data=new_df,hue='season')
plt.show()
```

### 年和季节的组合特征

```
def new_feature_year_season(train ,test):
    for X in [train,test]:
        X['year_season'] = X['year'] + 0.1 * X['season']
        
    by_season = train.groupby('year_season')[['count']].median()
    by_season.columns = ['count_year_season']
    
    train = train.join(by_season, on='year_season')
    test = test.join(by_season,on='year_season')
    
    return [train,test]
```

### 工作日高峰时期的标记特征
```
def new_feature_workingday_hour(X):
    X['hour_workingday_casual'] = X[['workingday','hour']].apply(
        lambda x : int(10 <= x['hour'] <= 19)
    ,axis=1)
    X['hour_workingday_registered'] = X[['workingday','hour']].apply(
        lambda x : int(
            (x['workingday'] == 1 and (x['hour'] == 8 or 17 <= x['hour'] <= 18))
            or (x['workingday'] == 0 and 10 <= x['hour'] <= 19 )
        )
    ,axis=1)
    return X
```



## 7) 关于调参以及StratifiedKFold

首先，交叉验证部分要用 StratifiedKFold

StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。

用法:

```
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
kf.split(X_train,year_month)
```

```
def plot_cv(param, bestreg, variable,title):
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(param[variable],bestreg.cv_results_['mean_test_score'],'o-')
    plt.xlabel(variable)
    plt.ylabel("score mean")
    plt.subplot(122)
    plt.plot(param[variable],bestreg.cv_results_['std_test_score'],'o-')
    plt.xlabel(variable)
    plt.ylabel("score std")
    plt.tight_layout()
    plt.title(title)
    plt.show()
```

```
def plot_rf():
    train ,_ ,_ ,_ = get_data()

    year_month = train['year'] * 100 + train['month']
    
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    common_columns = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'year', 'month', 'hour', 'dayofweek'
        ,'count_year_season'
    ]
    casual_columns = common_columns.copy()
    casual_columns.append('hour_workingday_casual')
    registered_columns = common_columns.copy()
    registered_columns.append('hour_workingday_registered')
    
    # rf n_estimators for casual and registered
    reg = RandomForestRegressor(random_state=0, n_jobs=-1)
    param = {"n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]}
    bestreg = GridSearchCV(reg, param, cv=kf.split(train, year_month), scoring=NMS)
    bestreg.fit(train[casual_columns], train['casual_log'])
    print(bestreg.best_params_)
    plot_cv(param, bestreg, "n_estimators",'rf casual')

    reg = RandomForestRegressor(random_state=0, n_jobs=-1)
    param = {"n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]}
    bestreg = GridSearchCV(reg, param, cv=kf.split(train, year_month), scoring=NMS)
    bestreg.fit(train[registered_columns], train['registered_log'])
    print(bestreg.best_params_)
    plot_cv(param, bestreg, "n_estimators",'rf registered')
    
    # rf tune the min_sample_split
    reg = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    param = {"min_samples_leaf": np.arange(1, 10, 1)}
    bestreg = GridSearchCV(reg, param, cv=kf.split(train, year_month), scoring=NMS)
    bestreg.fit(train[casual_columns], train['casual_log'])
    print(bestreg.best_params_)
    plot_cv(param, bestreg, "min_samples_leaf",'rf casual')

    reg = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    param = {"min_samples_leaf": np.arange(1, 10, 1)}
    bestreg = GridSearchCV(reg, param, cv=kf.split(train, year_month), scoring=NMS)
    bestreg.fit(train[registered_columns], train['registered_log'])
    print(bestreg.best_params_)
    plot_cv(param, bestreg, "min_samples_leaf",'rf registered')
    
    
plot_rf()
```

## 8) 此数据通过其他的回归方式无法获得正确的结果(预测结果中有负值，此会让RMSLE增大很多)

如何解决这个问题，就是将训练数据集中的看似无用的casual register两个变量分别提取出来，
首先经过大量的EDA:分析出casual和register之间的关系，和他们与target的关系如：

```
def plot_hour_hue_workingday():
    fig = plt.figure(figsize=(16,12))
    axes = fig.subplots(3,1)
    sns.boxplot(x = 'hour' ,y = 'count' ,hue = 'workingday' ,data = X_train ,ax = axes[0])
    sns.boxplot(x = 'hour' ,y = 'casual' ,hue = 'workingday' ,data = X_train ,ax = axes[1])
    sns.boxplot(x = 'hour' ,y = 'registered' ,hue = 'workingday' ,data = X_train ,ax = axes[2])
plot_hour_hue_workingday()
```

```
def plot_year_season(X):
    X['year_season'] = X['year'] + 0.1 * X['season']
    fig = plt.figure(figsize=(16,12))
    axes = fig.subplots(3,1)
    sns.boxplot(x = 'year_season' ,y = 'count' ,data = X ,ax = axes[0])
    sns.boxplot(x = 'year_season' ,y = 'casual' ,data = X ,ax = axes[1])
    sns.boxplot(x = 'year_season' ,y = 'registered' ,data = X ,ax = axes[2])
    
plot_year_season(X_train)
```

1. 生成 年份.季节 特征，因为随着此特征的变大，对于target有显著的正相关关系
2. 对casual进行标记 不管是workingday or not , 将casual对count有显著影响的部分标记出来，即 10:00 - 19:00 期间
3. 对register进行标记，常识判断register是工作者，会在工作日的[8:00,9:00]和[17:00,19:00]这个期间对count造成显著影响，而非工作日则和casual一致。
4. 通过这样的标记可以制造出三个与count有显著相关关系的特征，极大的增强了预测的准确率
5. 分别用casual, register这两个值作为目标值，以及不同的特征[casual or register]进行训练，将两个模型的预测值加在一起。


```
def pred_cv_xgb():
    train,y_trains,_,_ = get_data()
    y_train_count,y_trains_casual,y_trains_registered = y_trains[0],y_trains[1],y_trains[2]
    year_month = train['year'] * 100 + train['month']
    common_columns = [
        'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'year', 'month', 'hour', 'dayofweek'
        ,'count_year_season'
    ]
    casual_columns = common_columns.copy()
    casual_columns.append('hour_workingday_casual')
    registered_columns = common_columns.copy()
    registered_columns.append('hour_workingday_registered')
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    
    results1, results2 = [], []
    results3, results4, results5, results6 = [], [], [], []
    
    for train_ind, test_ind in kf.split(train, year_month):
        cur_train = train.iloc[train_ind,:]
        cur_test = train.iloc[test_ind,:]
        y_train_test = y_train_count[test_ind]
        
        # XGBREG 
        # casual
        reg = XGBRegressor(n_estimators=1000, gamma=1, random_state=0,n_jobs=-1)
        reg.fit(cur_train[casual_columns], cur_train['casual_log'])
        pred_casual = reg.predict(cur_test[casual_columns])
        pred_casual = np.exp(pred_casual) - 1
        pred_casual[pred_casual < 0] = 0
        
        # registered
        reg = XGBRegressor(n_estimators=1000, gamma=1, random_state=0,n_jobs=-1)
        reg.fit(cur_train[registered_columns], cur_train['registered_log'])
        pred_registered = reg.predict(cur_test[registered_columns])
        pred_registered = np.exp(pred_registered) - 1
        pred_registered[pred_registered < 0] = 0
        
        pred1 = pred_casual + pred_registered
        results1.append(RMLSE(y_train_test, pred1))

        # XGBRFREG
        # casual
        reg = XGBRFRegressor(n_estimators=50, gamma=1, random_state=0, n_jobs=-1)
        reg.fit(cur_train[casual_columns], cur_train['casual_log'])
        pred_casual = reg.predict(cur_test[casual_columns])
        pred_casual = np.exp(pred_casual) - 1
        pred_casual[pred_casual < 0] = 0

        # registered
        reg = XGBRFRegressor(n_estimators=50, gamma=1, random_state=0, n_jobs=-1)
        reg.fit(cur_train[registered_columns], cur_train['registered_log'])
        pred_registered = reg.predict(cur_test[registered_columns])
        pred_registered = np.exp(pred_registered) - 1
        pred_registered[pred_registered < 0] = 0
        pred2 = pred_casual + pred_registered
        results2.append(RMLSE(y_train_test, pred2))

        # see the two modules together
        pred_55 = 0.5 * pred1 + 0.5 * pred2
        results3.append(RMLSE(y_train_test, pred_55))
        pred_64 = 0.6 * pred1 + 0.4 * pred2
        results4.append(RMLSE(y_train_test, pred_64))
        pred_73 = 0.7 * pred1 + 0.3 * pred2
        results5.append(RMLSE(y_train_test, pred_73))
        pred_82 = 0.8 * pred1 + 0.2 * pred2
        results6.append(RMLSE(y_train_test, pred_82))

    print("XGB", np.mean(results1))
    print("XGBRF", np.mean(results2))
    print("0.5 * XGB + 0.5 * XGBRF", np.mean(results3))
    print("0.6 * XGB + 0.4 * XGBRF", np.mean(results4))
    print("0.7 * XGB + 0.3 * XGBRF", np.mean(results5))
    print("0.8 * XGB + 0.2 * XGBRF", np.mean(results6))
    
    return pred1,pred2,pred_55,pred_64,pred_73,pred_82

```

# 9) 加权的模型融合提高准确度

加权的模型融合：

```
score = 0.5 * GBDT + 0.5 * RF
score = w_1 * GBDT + w_2 * RF
```

可用线性回归得到最小值
