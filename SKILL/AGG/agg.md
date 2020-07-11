## groupBy size
```
agg = train.groupby(['groupId']).size().to_frame('players_in_team')
train = train.merge(agg, how='left', on=['groupId'])
```
## 以某个指标对某个特征变量进行排名
```
all_data['weaponsAcquired'] = all_data.groupby('matchId')['weaponsAcquired'].rank(pct=True).values
```

## 以某个指标对某个特征求均值
```
 meanData = all_data.groupby(['matchId','groupId'])[features].agg('mean')

```