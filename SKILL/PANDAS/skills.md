# 显示df的内存用量
```python
# 以m的方式
df.memory_usage().sum() / 1024**2
```

# 选择n个样本

# 显示df中isnull的对应特征的样本总数
```python
null_cnt = train.isnull().sum().sort_values()
```

# 数值类型数据的describe
```python
train.describe(include=np.number).T#.drop('count').T
train.describe(include=np.number).drop('count').T
```

# 特征变量中 nunique 值的数量
```python
for c in ['Id','groupId','matchId']:
    print(f'unique [{c}] count:', train[c].nunique())
```

# 简单的对单变量做value_counts 以及eda, 以及用lambda进行转换.
```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[0])

'''
solo  <-- solo,solo-fpp,normal-solo,normal-solo-fpp
duo   <-- duo,duo-fpp,normal-duo,normal-duo-fpp,crashfpp,crashtpp
squad <-- squad,squad-fpp,normal-squad,normal-squad-fpp,flarefpp,flaretpp
'''
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
train['matchType'] = train['matchType'].apply(mapper)
train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[1])
```

# query 查询 操作
```python
for q in ['numGroups == maxPlace','numGroups < maxPlace', 'numGroups > maxPlace']:
    print(q, ':', len(train.query(q)))
```

# groupby(cols)[cols2].describe()[叉乘multiindex]
```python
cols = ['numGroups','maxPlace']
desc1 = train.groupby('matchType')[cols].describe()[toTapleList(cols,['min','mean','max'])]
desc1
```

# groups in match size and count
```python

train.groupby(
    ['matchType','matchId','groupId']
    # 先以 matchType -> matchId -> groupId 为分组计算所有的count，这样能够将其他多余的以groupId为单位进行聚合
).size()

train.groupby(
    ['matchType','matchId','groupId']
    # 先以 matchType->matchId->groupId为分组计算所有的count，这样能够将其他多余的以groupId为单位进行聚合
).size().groupby(
    ['matchType','matchId']
    # 再聚合matchType->matchId
).size() # 再以matchType->matchId为单位聚合 计算size

```

# 多重复杂聚合的实例
```
cols = ['numGroups','maxPlace']

# describe
desc1 = train.groupby('matchType')[cols].describe()[toTapleList(cols,['min','mean','max'])]

# groups in match
group = train.groupby(
    ['matchType','matchId','groupId']
).count().groupby(
    ['matchType','matchId']
).size().to_frame(
    'groups in match'
)
desc2 = group.groupby('matchType').describe()[
    toTapleList(['groups in match'],['min','mean','max'])
]
pd.concat([desc1, desc2], axis=1)
```

# 复合复合describe
```
match = train.groupby(['matchType','matchId']).size().to_frame('players in match')
group = train.groupby(['matchType','matchId','groupId']).size().to_frame('players in group')
pd.concat([match.groupby('matchType').describe()[toTapleList(['players in match'],['min','mean','max'])], 
           group.groupby('matchType').describe()[toTapleList(['players in group'],['min','mean','max'])]], axis=1)
```

# loc转换
```
group = train.groupby(['matchId','groupId','matchType'])['Id'].count().to_frame('players').reset_index()
group.loc[group['players'] > 4, 'players'] = '5+'
group['players'] = group['players'].astype(str)
```

# 反向选择
```
sub = train.loc[~train['matchType'].str.contains('solo'),['winPlacePerc',col]].copy()

```

# rank as percent 相当于 将一个特征max min 归一化
```
match = all_data.groupby('matchId')
all_data['killsPerc'] = match['kills'].rank(pct=True).values
all_data['killPlacePerc'] = match['killPlace'].rank(pct=True).values
all_data['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values
#all_data['damageDealtPerc'] = match['damageDealt'].rank(pct=True).values
all_data['walkPerc_killsPerc'] = all_data['walkDistancePerc'] / all_data['killsPerc']
```

# fill inf 的值
```
def fillInf(df, val):
    numcols = df.select_dtypes(include='number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols: 
        df[c].fillna(val, inplace=True)
```

# reload 去掉不符合条件的match id.
```
def reload():
    gc.collect()
    df = pd.read_csv('../data/train_V2.csv')
    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
    df = df[-df['matchId'].isin(invalid_match_ids)]
    return df
```

# 获取随机n个样本

```python
def get_sample(df,n):
    """ 
    Gets a random sample of n rows from df, without replacement.
    Parameters:
    -----------
    df: A pandas data frame, that you wish to sample from.
    n: The number of rows you wish to sample.
    Returns:
    --------
    return value: A random sample of n rows of df.
    Examples:
    ---------
    >>> df = pd.DataFrame({'col1' : [1, 2, 3], 'col2' : ['a', 'b', 'a']})
    >>> df
       col1 col2
    0     1    a
    1     2    b
    2     3    a
    >>> get_sample(df, 2)
       col1 col2
    2     3    a
    1     2    b
    """
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()
```

# 显示头和尾的数据
```python
# First five rows (From Head)
print('First 5 rows: ')
display(train.head())

# Last five rows (To Tail)
print('Last 5 rows: ')
display(train.tail())
```

# 数据概览 describe() info()
```python
# Stats
train.describe().T

# Types, Data points, memory usage, etc.
train.info()

# Check dataframe's shape
print('Shape of training set: ', train.shape)
```

# 检查一个有缺失值的特征中对应的缺失值样本
```python
# Check row with NaN value
train[train['winPlacePerc'].isnull()]
```

# transform('count') & size()
```python
train.groupby('matchId')['matchId'].transform('count')
train.groupby('matchId')['matchId'].size()
```

# 利用query获得结果添加布尔值特征
```
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
```

# 将hashcode类型的特征转换为categoryid
```python
# Turn groupId and match Id into categorical types
train['groupId'] = train['groupId'].astype('category')
train['matchId'] = train['matchId'].astype('category')

# Get category coding for groupId and matchID
train['groupId_cat'] = train['groupId'].cat.codes
train['matchId_cat'] = train['matchId'].cat.codes

# Get rid of old columns
train.drop(columns=['groupId', 'matchId'], inplace=True)

# Lets take a look at our newly created features
train[['groupId_cat', 'matchId_cat']].head()
```

# 将特征总的类别变量转换成code
```python
    all_data['matchType'] = all_data['matchType'].map({
    'crashfpp':1,
    'crashtpp':2,
    'duo':3,
    'duo-fpp':4,
    'flarefpp':5,
    'flaretpp':6,
    'normal-duo':7,
    'normal-duo-fpp':8,
    'normal-solo':9,
    'normal-solo-fpp':10,
    'normal-squad':11,
    'normal-squad-fpp':12,
    'solo':13,
    'solo-fpp':14,
    'squad':15,
    'squad-fpp':16
    })
```

# 典型的聚合操作 groupby size resetindex mege
```python
matchSizeData = train.groupby(['matchId']).size() # 生成一个Series
# To specify the name of the new column use `name`.
# 不能inplace reset_index ，reset_index返回并创建一个DataFrame
matchSizeData  = matchSizeData.reset_index(name='matchSize')
#将聚合出来的那一列给上列名
#以左面的数据的key为基准给与新的特征
all_data = pd.merge(all_data, matchSizeData, how='left', on=['matchId'])
```