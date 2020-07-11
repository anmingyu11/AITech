# 忽略警告信息
```
# Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
```

# 计算叉积

```
from itertools import product
product(a,b,c,d)
```

# 反序

```python
cnt_srs.index[::-1]
```

# 判断纯数字或纯中文

```
a = 'aaaa'
a.isdigit()
print(a.encode('utf-8').isalpha())
print(a.isalpha())
```

```
c = '沃日'

print(c.encode('utf-8').isalpha())
print(c.isalpha())
```

https://blog.csdn.net/Refrain__WG/article/details/89214660

# 判断/删除空白字符

https://python3-cookbook.readthedocs.io/zh_CN/latest/c02/p11_strip_unwanted_characters.html

https://blog.csdn.net/qingsong3333/article/details/80350213

# 字典

https://juejin.im/post/5cacb76f6fb9a0685a3ee5de