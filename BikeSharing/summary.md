## 原特征
```
'datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count'
```

## 特征工程之后的
```
'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
'humidity', 'windspeed', 'year', 'month', 'hour', 'dayofweek',
'count_year_season'
```

# 异常值
19世纪俄国数学家切比雪夫研究统计规律中，论证并用标准差表达了一个不等式，这个不等式具有普遍的意义，被称作切比雪夫定理，其大意是：
任意一个数据集中，位于其平均数m个标准差范围内的比例（或部分）总是至少为1－1/(m^2)，其中m为大于1的任意正数。对于m=2，m=3和m=5有如下结果：
所有数据中，至少有3/4（或75%）的数据位于平均数2个标准差范围内。
所有数据中，至少有8/9（或88.9%）的数据位于平均数3个标准差范围内。
所有数据中，至少有24/25（或96%)的数据位于平均数5个标准差范围内

# 新特征

- count,casual,registered,作为连续型特征，特征变量的分布整体左倾，对其求对数使得数据正态化，提升模型效果。
- datetime -> year, month ,hour, dayofweek(周几)
- `year + 0.1 * season`，可以用于分层的交叉验证,也可以eda
- 对于casual的数量较多的时间 10:00-19:00
- 对于registered数量较多的时间 工作日：8:00-9:00,17:00-19:00, 非工作日10-19:00

# 融合
- 线性的模型融合 0.5GBDT+0.5RF



