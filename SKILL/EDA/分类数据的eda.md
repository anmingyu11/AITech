我们可以通过热力图来验证特征间的关系.
```
sns.swarmplot(x='hour',y='temp',data=new_df,hue='season')
plt.show()
```
