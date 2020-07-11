# int 类型的相关信息
```
np.iinfo(np.int8)
```

# float类型的相关信息
```
np.finfo(np.float32)
```

# np.random.permutation
> Randomly permute a sequence, or return a permuted range.

> If x is a multi-dimensional array, it is only shuffled along its first index.

```python
>>> np.random.permutation(10)
array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6])
>>> np.random.permutation([1, 4, 9, 12, 15])
array([15,  1,  9,  4, 12])
>>> arr = np.arange(9).reshape((3, 3))
>>> np.random.permutation(arr)
array([[6, 7, 8],
       [0, 1, 2],
       [3, 4, 5]])

```
