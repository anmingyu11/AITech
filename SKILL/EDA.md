# 单变量的countplot, sns提供的countplot是彩色的，较为优美.

```
def plot_bar(df,col):
    plt.figure(figsize=(12,4))
    sns.countplot(data=df,x=col).set_title('Kills')
    plt.show()
```

# [BikeSharing] 四级温度对应每个小时的分类曲线

```python
sns.swarmplot(x='hour',y='temp',data=new_df,hue='season')
plt.show()
```

# [BikeSharing] 移动平均 时间序列分析
```python
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

# [PUBG]画出两个特征之间的boxplot来看相关性
```
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='boosts', y="winPlacePerc", data=df_train)
fig.axis(ymin=0, ymax=1);
```

# [PUBG]画出两个特征之间的scatter来看相关性
```
df_train.plot(x="weaponsAcquired",y="winPlacePerc", kind="scatter", figsize = (8,6))
```

# [BikeSharing]FactorPlot 较为美观的countplot
```
sns.factorplot(x='holiday' ,data=train_df ,kind='count' ,size=5 ,aspect=1) # majority of data is for non holiday days.
```

# [BikeSharing] 多变量的 boxPlot
```
sns.boxplot(data=train_df[['temp','atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']])
fig = plt.gcf()
fig.set_size_inches(10,10
```

# [BikeSharing] 多个连续变量的直方图(切分成区间)
```
# can also be visulaized using histograms for all the continuous variables.
# 还可以使用直方图对所有连续变量进行可视化处理。

train_df.temp.unique()
fig , axes = plt.subplots(2,2)

axes[0,0].hist(x="temp" ,data=train_df ,edgecolor="black" ,linewidth=2 ,color='#ff4125')
axes[0,0].set_title("Variation of temp")

axes[0,1].hist(x="atemp" ,data=train_df ,edgecolor="black" ,linewidth=2 ,color='#ff4125')
axes[0,1].set_title("Variation of atemp")

axes[1,0].hist(x="windspeed" ,data=train_df ,edgecolor="black" ,linewidth=2 ,color='#ff4125')
axes[1,0].set_title("Variation of windspeed")

axes[1,1].hist(x="humidity" ,data=train_df ,edgecolor="black" ,linewidth=2 ,color='#ff4125')
axes[1,1].set_title("Variation of humidity")

fig.set_size_inches(10,10)
```
# [BikeSharing] 下三角的 heatmap
```
cor_mat= train_df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
```

```
corrMatt = dailyData[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
```

# [BikeSharing] factorplot 双变量barplot
```
sns.factorplot(x="hour",y="count",data=train_df,kind='bar',size=5,aspect=1.5)

sns.factorplot(x="day",y='count',kind='bar',data=train_df,size=5,aspect=1)
```

# [BikeSharing] 双变量 scatter
```
plt.scatter(x="temp",y="count",data=train_df,color='#ff4125')
```


# [BikeSharing] factorplot 横向的画出各个模型效果

```python
# 模型评估
models=[RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]
model_names=['RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']
rmsle=[]
d={}
for model in range (len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    test_pred=clf.predict(x_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_pred,y_test)))
d={'Modelling Algo':model_names,'RMSLE':rmsle}   
display(d)
# 生成DataFrame
rmsle_frame=pd.DataFrame(d)
display(rmsle_frame)
# 画出各个模型的效果
sns.factorplot(y='Modelling Algo',x='RMSLE',data=rmsle_frame,kind='bar',size=5,aspect=2)
```

# [BikeSharing] 折线图
```
sns.factorplot(x='Modelling Algo',y='RMSLE',data=rmsle_frame,kind='point',size=5,aspect=2)
```

# [BikeSharing] missingno缺失值分析
```
msno.matrix(dailyData,figsize=(12,5))
```

# [BikeSharing] 多个双变量的boxplot分析 (多图)

```
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sn.boxplot(data=dailyData,y="count",orient="v",ax=axes[0][0])
sn.boxplot(data=dailyData,y="count",x="season",orient="v",ax=axes[0][1])
sn.boxplot(data=dailyData,y="count",x="hour",orient="v",ax=axes[1][0])
sn.boxplot(data=dailyData,y="count",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Box Plot On Count")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")
axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
```

# [BikeSharing] 特征变量之间的散点图判断多重共线性

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

# [BikeSharing]三变量的eda

```
sns.swarmplot(x='hour',y='temp',data=new_df,hue='season')
plt.show()
```

# [PUBG] 双变量的scatter散点图分析相关性

```
df_train.plot(x="walkDistance",y="winPlacePerc", kind="scatter", figsize = (8,6))
```

# [PUBG] 双变量的box箱型图分析相关性

```
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='killStreaks', y="winPlacePerc", data=df_train)
#fig.axis(ymin=0, ymax=1);
```

# [PUBG] 双变量的散点图scatter jointplot分析相关性

```
sns.jointplot(x="winPlacePerc", y="kills", data=train, height=10, ratio=3, color="r")
plt.show()
```

# [PUBG] 双连续型变量的切分和boxplot分析相关性

```
kills = train.copy()

kills['killsCategories'] = pd.cut(
    kills['kills']
    , [-1, 0, 2, 5, 10, 60]
    , labels=['0_kills'
    ,'1-2_kills'
    , '3-5_kills'
    , '6-10_kills'
    , '10+_kills'])

plt.figure(figsize=(15,8))
sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)
plt.show()

del kills
gc.collect()
```

# [PUBG] 切比雪夫原则去除噪音并 绘制数据分布 distplot

简单的 0.99

```
data = train.copy()
data = data[data['walkDistance'] < train['walkDistance'].quantile(0.99)]
plt.figure(figsize=(15,10))
plt.title("Walking Distance Distribution",fontsize=15)
sns.distplot(data['walkDistance'])
plt.show()
```

# [PUBG] PointPlot 区间 画法PointPlot

```
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=train,color='#606060',alpha=0.8)
plt.xlabel('Number of Vehicle Destroys',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Vehicle Destroys/ Win Ratio',fontsize = 20,color='blue')
plt.grid()
plt.show()
```

# [PUBG] 双变量的pointplot

```
data = train.copy()
data = data[data['heals'] < data['heals'].quantile(0.99)]
data = data[data['boosts'] < data['boosts'].quantile(0.99)]

f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='heals',y='winPlacePerc',data=data,color='lime',alpha=0.8)
sns.pointplot(x='boosts',y='winPlacePerc',data=data,color='blue',alpha=0.8)
plt.text(4,0.6,'Heals',color='lime',fontsize = 17,style = 'italic')
plt.text(4,0.55,'Boosts',color='blue',fontsize = 17,style = 'italic')
plt.xlabel('Number of heal/boost items',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Heals vs Boosts',fontsize = 20,color='blue')
plt.grid()
plt.show()
del data
gc.collect()
```

# [PUBG] 三变量 数据切分 pointplot
切分部分:

```
solos = train[train['numGroups']>50]
duos = train[(train['numGroups']>25) & (train['numGroups']<=50)]
squads = train[train['numGroups']<=25]
print("There are {} ({:.2f}%) solo games, {} ({:.2f}%) duo games and {} ({:.2f}%) squad games."
      .format(len(solos), 100*len(solos)/len(train), len(duos), 100*len(duos)/len(train), len(squads), 100*len(squads)/len(train),))
```

pointplot 部分:

```
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='kills',y='winPlacePerc',data=solos,color='black',alpha=0.8)
sns.pointplot(x='kills',y='winPlacePerc',data=duos,color='#CC0000',alpha=0.8)
sns.pointplot(x='kills',y='winPlacePerc',data=squads,color='#3399FF',alpha=0.8)
plt.text(37,0.6,'Solos',color='black',fontsize = 17,style = 'italic')
plt.text(37,0.55,'Duos',color='#CC0000',fontsize = 17,style = 'italic')
plt.text(37,0.5,'Squads',color='#3399FF',fontsize = 17,style = 'italic')
plt.xlabel('Number of kills',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Solo vs Duo vs Squad Kills',fontsize = 20,color='blue')
plt.grid()
plt.show()
```

# [PUBG] 四变量 pointPlot

```
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='DBNOs',y='winPlacePerc',data=duos,color='#CC0000',alpha=0.8)
sns.pointplot(x='DBNOs',y='winPlacePerc',data=squads,color='#3399FF',alpha=0.8)
sns.pointplot(x='assists',y='winPlacePerc',data=duos,color='#FF6666',alpha=0.8)
sns.pointplot(x='assists',y='winPlacePerc',data=squads,color='#CCE5FF',alpha=0.8)
sns.pointplot(x='revives',y='winPlacePerc',data=duos,color='#660000',alpha=0.8)
sns.pointplot(x='revives',y='winPlacePerc',data=squads,color='#000066',alpha=0.8)
plt.text(14,0.5,'Duos - Assists',color='#FF6666',fontsize = 17,style = 'italic')
plt.text(14,0.45,'Duos - DBNOs',color='#CC0000',fontsize = 17,style = 'italic')
plt.text(14,0.4,'Duos - Revives',color='#660000',fontsize = 17,style = 'italic')
plt.text(14,0.35,'Squads - Assists',color='#CCE5FF',fontsize = 17,style = 'italic')
plt.text(14,0.3,'Squads - DBNOs',color='#3399FF',fontsize = 17,style = 'italic')
plt.text(14,0.25,'Squads - Revives',color='#000066',fontsize = 17,style = 'italic')
plt.xlabel('Number of DBNOs/Assits/Revives',fontsize = 15,color='blue')
plt.ylabel('Win Percentage',fontsize = 15,color='blue')
plt.title('Duo vs Squad DBNOs, Assists, and Revives',fontsize = 20,color='blue')
plt.grid()
plt.show()
```

# [PUBG] groupby.count countplot 很好看 

```
# playersJoined
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
plt.figure(figsize=(15,10))
sns.countplot(train[train['playersJoined']>=75]['playersJoined'])
plt.title('playersJoined')
plt.show()
```

# [PUBG] distplot 离散型变量或离散变量有大量值 切分bins

```
# Plot the distribution of weaponsAcquired
plt.figure(figsize=(12,4))
sns.distplot(train['weaponsAcquired'], bins=100)
plt.show()
```

# [SCTP] 所有特征之间两两散点图

```
def plot_feature_scatter(df1, df2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(4,4,figsize=(14,14))

    for feature in features:
        i += 1
        plt.subplot(4,4,i)
        plt.scatter(df1[feature], df2[feature], marker='+')
        plt.xlabel(feature, fontsize=9)
    plt.show();

# 使用
features = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7', 
           'var_8', 'var_9', 'var_10','var_11','var_12', 'var_13', 'var_14', 'var_15', 
           ]
plot_feature_scatter(train_df[::20],test_df[::20], features)
```

# [SCTP] df1[feature][label1] df2[feature][label2] 对应的distplot 数据分布,观察不同标签对应相同特征的分布


```
def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();
    
# 使用
t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[2:102]
plot_feature_distribution(t0, t1, '0', '1', features)
```

# [SCTP] 计算每个样本所有特征变量的平均值，画出训练集和测试集的数据分布比较区别(std,min,max,skew...)

```
plt.figure(figsize=(16,6))
features = train_df.columns.values[2:202]
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train_df[features].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=1),color="blue", kde=True,bins=120, label='test')
plt.legend()
plt.show()
```

# [SCTP] 计算一个样本集合中每个特征的平均值，画出训练集与测试集的区别(std,min,max,skew...)

```python
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train and test set")
sns.distplot(train_df[features].mean(axis=0),color="magenta",kde=True,bins=120, label='train')
sns.distplot(test_df[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='test')
plt.legend()
plt.show()
```

# [SCTP] 计算二分类数据中两个分类每个样本对应的所有特征的均值,和所有特征对应的均值(std,min,max,skew...)

```
t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train set")
sns.distplot(t0[features].mean(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend()
plt.show()
```

```
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per column in the train set")
sns.distplot(t0[features].mean(axis=0),color="green", kde=True,bins=120, label='target = 0')
sns.distplot(t1[features].mean(axis=0),color="darkblue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
```

# [SCTP] countplot pieplot 画出二分类 target的 饼状图和countplot

```
f,ax=plt.subplots(1,2,figsize=(18,8))
train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('target',data=train,ax=ax[1])
ax[1].set_title('target')
plt.show()
```

# 解决中文字体问题

https://www.zhihu.com/question/25404709

# [PFS] 画出每个商店对应的时间序列的售出数量

```
grouped = pd.DataFrame(
    train.groupby(
        ['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
num_graph = 10
id_per_graph = ceil(grouped.shop_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
        count += 1
```


# [PFS] 80种类别对应的月销量

```
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
num_graph = 10
id_per_graph = ceil(train.item_category_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='item_category_id', 
                      data=train[np.logical_and(count*id_per_graph <= train['item_category_id'], train['item_category_id'] < (count+1)*id_per_graph)], 
                      ax=axes[i][j])
        count += 1
```

# [PFS] boxplot 分析异常值

```
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=sale_train['item_cnt_day'])
print('Sale volume outliers:',sale_train['item_id'][sale_train['item_cnt_day']>500].unique())

plt.figure(figsize=(10,4))
#plt.xlim(-100,  sale_train['item_price'].max())
#plt.xlim(sale_train['item_price'].min(), sale_train['item_price'].max())
sns.boxplot(x=sale_train['item_price'])
print('Item price outliers:',sale_train['item_id'][sale_train['item_price']>50000].unique())
```


# [PFS] ploty 跟时间序列有关的图像

```
daily_sales_sc = go.Scatter(x=daily_sales['date'], y=daily_sales['sales'])
layout = go.Layout(title='Daily sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
fig = go.Figure(data=[daily_sales_sc], layout=layout)
iplot(fig)
```


```
store_daily_sales_sc = []
for store in store_daily_sales['store'].unique():
    current_store_daily_sales = store_daily_sales[(store_daily_sales['store'] == store)]
    store_daily_sales_sc.append(
        go.Scatter(
            x=current_store_daily_sales['date']
            , y=current_store_daily_sales['sales']
            , name=('Store %s' % store)))

layout = go.Layout(title='Store daily sales', xaxis=dict(title='Date'), yaxis=dict(title='Sales'))
fig = go.Figure(data=store_daily_sales_sc, layout=layout)
iplot(fig)
```

# [elo] boxplot 比较特征间的关系

```python
# feature 1
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_1", y=target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 1 distribution")
plt.show()

# feature 2
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_2", y=target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 2', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 2 distribution")
plt.show()

# feature 3
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_3", y=target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 3', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 3 distribution")
plt.show()
```

# [elo] iplot的折线图，非常优美

```python
cnt_srs = train_df.groupby("num_hist_transactions")[target_col].mean()
cnt_srs = cnt_srs.sort_index()
cnt_srs = cnt_srs[:-50]

def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

trace = scatter_plot(cnt_srs, "orange")
layout = dict(
    title='Loyalty score by Number of historical transactions',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Histtranscnt")
```

# [elo] 双离散变量的关联分析 箱型图

```python
bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 10000]
train_df['binned_num_hist_transactions'] = pd.cut(train_df['num_hist_transactions'], bins)
cnt_srs = train_df.groupby("binned_num_hist_transactions")[target_col].mean()

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_num_hist_transactions", y=target_col, data=train_df, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_num_hist_transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("binned_num_hist_transactions distribution")
plt.show()
```

# 饼状图

```python
def plot_pie(x):
    labels,counts = np.unique(x , return_counts=True)
    fig = plt.figure(figsize=(8,6))
    plt.pie(counts,labels=labels,autopct='%1.2f%%') #画饼图（数据，数据对应的标签，百分数保留两位小数点）
    plt.title("Pie chart")
    plt.show()
```

