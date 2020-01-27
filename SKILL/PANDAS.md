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
    X_date = pd.DatetimeIndex(X['datetime'])
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