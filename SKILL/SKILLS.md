# [PUBG]减小数据占有内存的用量

```
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```

# [PUBG]计算两个列表的叉积 

```
def toTapleList(list1,list2):
    return list(itertools.product(list1,list2))
```
# [PUBG]切分数据，以某个离散特征单独切分

### 1. 

```
print("Split time")
def split_train_val(data, fraction):
    # 以matchd为标准对数据进行切分
    matchIds = data['matchId'].unique().reshape([-1])
    train_size = int(len(matchIds)*fraction)
    
    random_idx = np.random.RandomState(seed=2).permutation(len(matchIds))
    train_matchIds = matchIds[random_idx[:train_size]]
    val_matchIds = matchIds[random_idx[train_size:]]
    
    data_train = data.loc[data['matchId'].isin(train_matchIds)]
    data_val = data.loc[data['matchId'].isin(val_matchIds)]
    return data_train, data_val

# Split the Data by matchId. Thanks to Ivan Batalov for this. 
# 将train拆分成两部分,一部分是训练数据，一部分是用来验证模型效果， 只用训练数据来做拆分
X_train, X_train_test = split_train_val(X_train, 0.91)
print("Y time")
y = X_train['winPlacePerc']
y_test = X_train_test['winPlacePerc']
print("X_train time")
X_train = X_train.drop(columns=['matchId', 'winPlacePerc'])
print("X test train time")
X_train_test = X_train_test.drop(columns='matchId')
print("X test train winPlace remove")
X_train_test = X_train_test.drop(columns='winPlacePerc')

print("X test np time")
X_train_test = np.array(X_train_test)
print("y test np time")
y_test = np.array(y_test)
```

### 2.

```
def train_test_split(df,test_size=0.1):
    match_ids = df['matchId'].unique().tolist()
    train_size = int(len(match_ids) * (1 - test_size))
    train_match_ids = random.sample(match_ids,train_size)
    
    train = df[df['matchId'].isin(train_match_ids)]# isin
    test = df[-df['matchId'].isin(train_match_ids)]
    
    return train,test
```

# [PFS] 先看submission

```
# for kaggle competition, always look at sample_submission.csv first, so you know what you want to get
# then train.csv and test.csv
sub = pd.read_csv('../data/sample_submission.csv')
sub.head()
```

# [PFS] 运行外部命令

```
!ls ../data/*
```