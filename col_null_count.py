def col_null_count(df):
    train_dense_missing = df.isnull().sum(axis=0)
    for i in range(0, len(train_dense_missing)):
        if train_dense_missing[i] > 0:
            print(train_dense_missing.index[i] + ' : ' + str(train_dense_missing[i]))
    print('\n')
    return
