```
season=pd.get_dummies(train_df['season'],prefix='season')
df=pd.concat([train_df,season],axis=1)
display(train_df.head())
season=pd.get_dummies(test_df['season'],prefix='season')
test_df=pd.concat([test_df,season],axis=1)
display(test_df.head())
```
