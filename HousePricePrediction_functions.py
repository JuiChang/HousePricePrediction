import math
import pandas as pd


def cat_to_num(cat_list, ser):
    print("In")
    for i in range(0, len(ser)):
        # print(i)
        for j in range(0, len(cat_list)):
            # print(ser[i])
            if ser[i] == cat_list[j]:
                # ser.iat[i] = j + 1
                ser.iat[i] = len(cat_list) - j
                break
            elif cat_list[j] == 'NA':
                if math.isnan(ser.iat[i]):
                    ser.iat[i] = len(cat_list) - j
                    break
    return


def col_null_count(df):
    train_dense_missing = df.isnull().sum(axis=0)
    for i in range(0, len(train_dense_missing)):
        if train_dense_missing[i] > 0:
            print(train_dense_missing.index[i] + ' : ' + str(train_dense_missing[i]))
    print('\n')
    return


def rm_sparse_row_col(train_conv, test_conv, train_y,
                      # train_dense, test_follow_dense, train_y_dense_row,
                      row_thres, col_thres):
    ret = list()
    sparse_row = list()
    for row in range(0, len(train_conv)):
        if train_conv.loc[row].isnull().sum() > len(train_conv.columns)*row_thres:
            sparse_row.append(row)
    train_dense_row = train_conv.drop(train_conv.index[sparse_row])
    # train_y_dense_row = train_y.drop(train_y.index[sparse_row])
    ret.append(train_y.drop(train_y.index[sparse_row]))

    dense_column = list()
    for col in range(0, len(train_dense_row.columns)):
        if train_dense_row.iloc[:, col].isnull().sum() < len(train_conv)/(1-col_thres):
            dense_column.append(col)
    # train_dense = train_dense_row.iloc[:, dense_column]
    ret.append(train_dense_row.iloc[:, dense_column])
    # test_follow_dense = test_conv.iloc[:, dense_column]
    ret.append(test_conv.iloc[:, dense_column])

    return ret


def impute_mode_aver(combine_impute, cate_vari_list, numer_vari_list):
    for colname in cate_vari_list:
        if colname in list(combine_impute.columns.values):
            if combine_impute[colname].isnull().sum() > 0:
                null_list = combine_impute[colname].isnull()
                mode = combine_impute[colname].mode().values[0]
                for index in range(0, len(combine_impute)):
                    if null_list[index]:
                        combine_impute[colname].iat[index] = mode

    for colname in numer_vari_list:
        if colname in list(combine_impute.columns.values):
            if combine_impute[colname].isnull().sum() > 0:
                null_list = combine_impute[colname].isnull()
                mean = combine_impute[colname].mean()
                for index in range(0, len(combine_impute)):
                    if null_list[index]:
                        combine_impute[colname].iat[index] = mean
    return


def create_dum_vari(df_in_dum, cate_vari_list):
    df_dum = df_in_dum.copy()
    for colname in cate_vari_list:
        if colname in list(df_dum.columns.values):
            df_dum = pd.concat([df_dum, pd.get_dummies(df_dum[colname])], axis=1)
            df_dum.drop([colname], axis=1, inplace=True)
    return df_dum
