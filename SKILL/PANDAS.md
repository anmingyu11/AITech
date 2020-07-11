# [BikeSharing] one-hot
## 加前缀的one_hot
```
season=pd.get_dummies(train_df['season'],prefix='season')
```

# [BikeSharing] 丢弃某个特征
```
train_df.drop(['season','weather'],inplace=True,axis=1)
display(train_df.head())
```

# [BikeSharing] 时间戳分解
```
train_df["hour"] = [t.hour for t in pd.DatetimeIndex(train_df.datetime)]
train_df["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_df.datetime)]
train_df["month"] = [t.month for t in pd.DatetimeIndex(train_df.datetime)]
train_df['year'] = [t.year for t in pd.DatetimeIndex(train_df.datetime)]
train_df['year'] = train_df['year'].map({2011:0, 2012:1})
train_df.head()
```

```
def translate_datetime(X):
    X_ = X.copy()
    X_date = 
    X_['year'] = X_date.year
    X_['month'] = X_date.month
    X_['hour'] = X_date.hour
    X_['dayofweek'] = X_date.dayofweek
    return X_
```

# [BikeSharing] 对数据列变量类型的概览
```
train_df.columns.to_series().groupby(train_df.dtypes).groups
```

# [PUBG] 聚合操作

### groupby.size
```
agg = df.groupby(['groupId']).size().to_frame('players_in_team')

# 统计groupId对应的数量
```

### merge

```
df = df.merge(agg , how='left' , on=['groupId'])
# 利用merge来将 groupId对应的players in team 数量合并到表中.
```

# min,max,sum,median,mean,rank

先以matchId,再以groupId为单位，分别计算min,max,sum,median,rank

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

# [PUBG] 计算比例

```
df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
df['killPlace_over_maxPlace'].fillna(0,inplace=True)
df['killPlace_over_maxPlace'].replace(np.inf,0,inplace=True)
```

# [PUBG] 对单变量进行简略的分布输出
```
print("The average person uses {:.1f} heal items, 99% of people use {} or less, while the doctor used {}."
      .format(train['heals'].mean(), train['heals'].quantile(0.99), train['heals'].max()))
print("The average person uses {:.1f} boost items, 99% of people use {} or less, while the doctor used {}."
      .format(train['boosts'].mean(), train['boosts'].quantile(0.99), train['boosts'].max()))
```

# [PUBG] describe 这样写输出更优美

```
train.describe().T
train.info()
```

# [PUBG] 复合的布尔操作

```
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
```

# [PUBG] pandas dataframe.agg操作

**agg (map {key : 对应的特征名称, val : 对应的聚合操作})**

agg (map)

```
agg = train.groupby(['matchId']).agg({'players_in_team':['min','max','mean']})
display(agg)
display(agg.columns)
display(agg.columns.ravel())
```

agg后合并到dataframe

```
agg = train.groupby(['matchId']).agg({'players_in_team': ['min', 'max', 'mean']})
agg.columns = ['_'.join(x) for x in agg.columns.ravel()]
train = train.merge(agg, how='left', on=['matchId'])
train['players_in_team_var'] = train.groupby(['matchId'])['players_in_team'].var()
display(train[['matchId', 'players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']].head())
display(train[['matchId', 'players_in_team', 'players_in_team_min', 'players_in_team_max', 'players_in_team_mean', 'players_in_team_var']].describe())
```

pandas document:

```
agg is an alias for aggregate. Use the alias.

A passed user-defined-function will be passed a Series for evaluation.

Examples

>>> df = pd.DataFrame([[1, 2, 3],
...                    [4, 5, 6],
...                    [7, 8, 9],
...                    [np.nan, np.nan, np.nan]],
...                   columns=['A', 'B', 'C'])
Aggregate these functions over the rows.

>>> df.agg(['sum', 'min'])
        A     B     C
sum  12.0  15.0  18.0
min   1.0   2.0   3.0
Different aggregations per column.

>>> df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})
        A    B
max   NaN  8.0
min   1.0  2.0
sum  12.0  NaN
Aggregate over the columns.

>>> df.agg("mean", axis="columns")
0    2.0
1    5.0
2    8.0
3    NaN
dtype: float64
```

# [PUBG] 查看数据的缺失值

```
null_cnt = train.isnull().sum().sort_values()
```

**Santander Customer Transaction**

```
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return tt,(np.transpose(tt))
```


# [PUBG] 填充无效无穷的值

```
def fillInf(df, val):
    numcols = df.select_dtypes(include='number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols: 
        df[c].fillna(val, inplace=True)
```

# [SCTP] 查看missing data

```
def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)
```

# PANDAS按条件删除行
Pandas怎样按条件删除行？

```
df = df.drop(some labels)
```
```
df = df.drop(df[].index)
```

Example

To remove all rows where column ‘score’ is < 50:

```
df = df.drop(df[df.score < 50].index)
```

In place version (as pointed out in comments)

```
df.drop(df[df.score < 50].index, inplace=True)
```

Multiple conditions

(see Boolean Indexing)

The operators are: | for or, & for and, and ~ for not. These must be grouped by using parentheses.

To remove all rows where column 'score' is < 50 and > 20

```
df = df.drop(df[(df.score < 50) & (df.score > 20)].index)
```

# 生成一个日期序列

```
import pandas as pd
from datetime import datetime
def datelist(beginDate, endDate):
    # beginDate, endDate是形如‘20160601’的字符串或datetime格式
    date_l=[datetime.strftime(x,'%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

datelist('2016-01-01','2020-01-01')
```

# 删除DataFrame中值全为NaN或者包含有NaN的列或行

### 删除NaN所在的行：

#### 删除表中全部为NaN的行

```
df.dropna(axis=0,how='all')  
```

#### 删除表中含有任何NaN的行

```
df.dropna(axis=0,how='any') #drop all rows that have any NaN values
```

### 删除NaN所在的列：

#### 删除表中全部为NaN的列

```
df.dropna(axis=1,how='all') 
```

#### 删除表中含有任何NaN的列

```
df.dropna(axis=1,how='any') #drop all rows that have any NaN values
```

# Shift

shift函数是对数据进行移动的操作，假如现在有一个DataFrame数据df，如下所示：

index | value1
--- | ---
A|0
B|1
C|2
D|3

那么如果执行以下代码：

```
df.shift()
```
就会变成如下：

index | value1
--- | ---
A|NaN
B|0
C|1
D|2

看一下函数原型：

```
DataFrame.shift(periods=1, freq=None, axis=0)
```

参数

`periods` ：类型为int，表示移动的幅度，可以是正数，也可以是负数，默认值是1,1就表示移动一次，注意这里移动的都是数据，而索引是不移动的，移动之后没有对应值的，就赋值为NaN。

执行以下代码：

```
df.shift(2)
```

就会得到：

index | value1
--- | ---
A|NaN
B|NaN
C|0
D|1

执行：

```
df.shift(-1)
```

会得到：

index|value1
--- | ---
A|1
B|2
C|3
D|NaN

freq： DateOffset, timedelta, or time rule string，

可选参数，默认值为None，只适用于时间序列，如果这个参数存在，那么会按照参数值移动时间索引，而数据值没有发生变化。例如现在有df1如下：

index | value1
--- | ---
2016-06-01 | 0
2016-06-02 | 1
2016-06-03 | 2
2016-06-04 | 3

执行：

```
df1.shift(periods=1,freq=datetime.timedelta(1))
```

会得到：

index | value1
—-|—-
2016-06-02 | 0
2016-06-03 | 1
2016-06-04 | 2
2016-06-05 | 3

```
axis：{0, 1, ‘index’, ‘columns’}:
```

表示移动的方向，如果是0或者’index’表示上下移动，如果是1或者’columns’，则会左右移动。

# div

示例 1: 

使用div()函数查找具有常量值的dataframe元素的商。还要处理dataframe中出现的NaN值。

```
# importing pandas as pd 
import pandas as pd 
  
# Creating the dataframe with NaN value 
df = pd.DataFrame({"A":[5, 3, None, 4], 
                   "B":[None, 2, 4, 3],  
                   "C":[4, 3, 8, 5], 
                   "D":[5, 4, 2, None]}) 
  
# Print the dataframe 
df 
```

![](https://media.geeksforgeeks.org/wp-content/uploads/1-472.png)

Now find the division of each dataframe element with 2

```
# Find the division with 50 being substituted 
# for all the missing values in the dataframe 
df.div(2, fill_value = 50) 
```

![](https://media.geeksforgeeks.org/wp-content/uploads/1-473.png)

The output is a dataframe with cells containing the result of the division of each cell value with 2.

All the NaN cells have been filled with 50 before performing the division.

Example 2 : Use div() function to find the floating division of a dataframe with a series object over the index axis.

```
# importing pandas as pd 
import pandas as pd 

# Creating the dataframe 
df = pd.DataFrame({"A":[5, 3, 6, 4], 
				"B":[11, 2, 4, 3], 
				"C":[4, 3, 8, 5], 
				"D":[5, 4, 2, 8]}) 

# Create a series object with no. of elements 
# equal to the element along the index axis. 

# Creating a pandas series object 
series_object = pd.Series([2, 3, 1.5, 4]) 

# Print the series_obejct 
series_object 
```

Output :

![](https://media.geeksforgeeks.org/wp-content/uploads/1-474.png)

Note: If the dimension of the index axis of the dataframe and the series object is not same then an error will occur.
> 注意 : 如果`dataframe`和`series`对象的索引轴的维数不相同，则会出现错误。

Now, find the division of dataframe elements with the series object along the index axis
> 现在，查找dataframe元素与系列对象在索引轴上的division

```
# To find the division 
df.div(series_object, axis = 0)
```

Output :

![](https://media.geeksforgeeks.org/wp-content/uploads/1-475.png)

The output is a dataframe with cells containing the result of the division of the current cell element with the corresponding series object cell.
> 输出是一个dataframe，其中的单元包含当前单元元素与相应的系列对象单元划分的结果。

# [PFS]Join的用法

与merge 很类似
```
train = train.set_index('item_id').join(items.set_index('item_id')).drop('item_name', axis=1).reset_index()
```

# [PFS] 复杂的聚合操作

从月份，到对应的商店，到对应的商品，到对应的日期和价格还有数量
```
monthly_sales=sales.groupby(
    ["date_block_num","shop_id","item_id"]
)["date","item_price","item_cnt_day"].agg(
    {"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"}
)
```

# [PFS] 每一类商品的数量聚类

```
# number of items per cat 
x = item.groupby(['item_category_id']).count()
x = x.sort_values(by='item_id',ascending=False)
x = x.iloc[0:10].reset_index()
x
```


# [PFS] 给列的重命名

```
train_clean = train_clean.rename(index = str, columns = {"item_cnt_day" : "item_cnt_month"})
```

# [PFS] 缺失月份补全

```
month_list=[i for i in range(num_month+1)]
shop = []
for i in range(num_month+1):
    shop.append(5)
item = []
for i in range(num_month+1):
    item.append(5037)
months_full = pd.DataFrame({'shop_id':shop, 'item_id':item,'date_block_num':month_list})
months_full
```

# [PFS] 查看重复值并显示

```
display(sale_train[sale_train.duplicated()])
print('Number of duplicates:', len(sale_train[sale_train.duplicated()]))
```

# [PFS] Pivot_table

> [一文看懂透视表](https://zhuanlan.zhihu.com/p/31952948)
> 

```
sales_by_item_id = sale_train.pivot_table(
    index=['item_id'] , values=['item_cnt_day'], 
    columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'
sales_by_item_id
```

# [PFS] 对行计算和

```
outdated_items = sales_by_item_id[sales_by_item_id.loc[:,'27':].sum(axis=1)==0]
print('Outdated items:', len(outdated_items))
```

# [PFS] 通过特征切分数据排除异常值这样更快

```python
train_monthly = train_monthly.query('item_cnt >= 0 and item_cnt <= 20 and item_price < 400000')
```

# [Elo] 聚合操作集合

```python
def aggregate_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
    'category_1': ['sum', 'mean'],
    'category_2_1.0': ['mean'],
    'category_2_2.0': ['mean'],
    'category_2_3.0': ['mean'],
    'category_2_4.0': ['mean'],
    'category_2_5.0': ['mean'],
    'category_3_A': ['mean'],
    'category_3_B': ['mean'],
    'category_3_C': ['mean'],
    'merchant_id': ['nunique'],
    'merchant_category_id': ['nunique'],
    'state_id': ['nunique'],
    'city_id': ['nunique'],
    'subsector_id': ['nunique'],
    'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
    'installments': ['sum', 'mean', 'max', 'min', 'std'],
    'purchase_month': ['mean', 'max', 'min', 'std'],
    'purchase_date': [np.ptp, 'min', 'max'],
    'month_lag': ['mean', 'max', 'min', 'std'],
    'month_diff': ['mean']
    }
    
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['_'.join(col).strip() for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history
```

# 时间，日期，索引

https://www.jianshu.com/p/4ece5843d383

# 插入行

https://www.jianshu.com/p/7df2593a01ce



# pandas中关于DataFrame行，列显示不完全（省略）的解决办法

https://blog.csdn.net/weekdawn/article/details/81389865

http://sofasofa.io/forum_main_post.php?postid=1000912