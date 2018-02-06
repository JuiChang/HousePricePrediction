import math
import pandas as pd
import numpy as np
import sys


def cat_to_num(cat_list, ser):
    # print("In")
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
                      row_thres, col_thres):
    ret = list()
    sparse_row = list()
    for row in range(0, len(train_conv)):
        if train_conv.loc[row].isnull().sum() > len(train_conv.columns)*row_thres:
            sparse_row.append(row)
    train_dense_row = train_conv.drop(train_conv.index[sparse_row])
    ret.append(train_y.drop(train_y.index[sparse_row]))

    dense_column = list()
    for col in range(0, len(train_dense_row.columns)):
        if train_dense_row.iloc[:, col].isnull().sum() < len(train_conv)*(1-col_thres):
            dense_column.append(col)
    ret.append(train_dense_row.iloc[:, dense_column])
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
    ret = list()
    df_dum = df_in_dum.copy()
    new_columns = list()
    for colname in cate_vari_list:
        # print('for start')
        if colname in list(df_dum.columns.values):
            tmp_df = pd.get_dummies(df_dum[colname])
            tmp_df = avoid_duplicate_label(df_dum, tmp_df)
            tmp_df = tmp_df.reset_index(drop=True)
            df_dum = pd.concat([df_dum, tmp_df], axis=1, join_axes=[df_dum.index])
            # df_dum = pd.concat([df_dum, tmp_df], axis=1)
            # print(list(tmp_df.columns.values)[0])
            # print(df_dum[list(tmp_df.columns.values)[0]])
            df_dum.drop([colname], axis=1, inplace=True)
            new_columns += list(tmp_df.columns.values)
    ret.append(df_dum)
    ret.append(new_columns)
    return ret


def avoid_duplicate_label(df_big_in, df_small_in):
    df_big = df_big_in.copy()
    df_small = df_small_in.copy()
    for col_name in list(df_small.columns.values):
        # print('col_name: ', col_name)
        tmp_col_name = col_name
        while tmp_col_name in list(df_big.columns.values):
            # df_small.rename(index=str, columns={tmp_col_name: '$'+tmp_col_name}, inplace=True)
            df_small = df_small.rename(index=str, columns={tmp_col_name: '$'+tmp_col_name})
            tmp_col_name = '$' + tmp_col_name
        # print('tmp_col_name: ', tmp_col_name)
    # print(df_small.iloc[:10, :])
    # print('before return')
    return df_small


def binary_feature_selection_by_averYdiff(df_trainX_in, trainY_in, df_testX_in, vari_list, threshold):
    ret = list()
    dict_aver_diff = dict()
    df_trainX = df_trainX_in.copy()
    df_testX = df_testX_in.copy()
    trainY = trainY_in.copy()
    drop_columns = list()

    # calculate the difference of the two averages of each columns
    for colname in list(df_trainX.columns.values):
        if colname in vari_list:
            # print(len(df_trainX))
            # print(colname)
            one_sum = 0
            one_count = 0
            zero_sum = 0
            zero_count = 0
            for i in range(0, len(df_trainX)):
                # print(i)
                if df_trainX[colname].iat[i] == 1:
                    one_sum += trainY.iat[i]
                    one_count += 1
                else:
                    zero_sum += trainY.iat[i]
                    zero_count += 1
            if one_count != 0 and zero_count != 0:
                dict_aver_diff[colname] = one_sum/one_count - zero_sum/zero_count
            else:
                dict_aver_diff[colname] = 0
    # print(dict_aver_diff.values())
    print(dict_aver_diff)
    # print(len(dict_aver_diff))

    # get the columns to drop
    for key in dict_aver_diff.keys():
        if abs(dict_aver_diff[key]) < threshold:
            drop_columns.append(key)

    # drop the columns
    df_trainX.drop(drop_columns, axis=1, inplace=True)
    df_testX.drop(drop_columns, axis=1, inplace=True)

    ret.append(df_trainX)
    ret.append(df_testX)
    return ret


# by correlation coefficient
def numerical_feature_selection_by_CC(df_trainX_in, trainY_in, df_testX_in, vari_list, threshold):
    ret = list()
    df_trainX = df_trainX_in.copy()
    df_testX = df_testX_in.copy()
    trainY = trainY_in.copy()
    dict_cc = dict()
    drop_columns = list()

    for colname in list(df_trainX.columns.values):
        if colname in vari_list:
            # print(colname)
            sys.stdout.flush()
            dict_cc[colname] = np.corrcoef(df_trainX[colname].values.astype(float),
                                           trainY.values)[1, 0]
    print(dict_cc)

    # get the columns to drop
    for key in dict_cc.keys():
        if abs(dict_cc[key]) < threshold:
            drop_columns.append(key)

    # drop the columns
    df_trainX.drop(drop_columns, axis=1, inplace=True)
    df_testX.drop(drop_columns, axis=1, inplace=True)

    ret.append(df_trainX)
    ret.append(df_testX)
    return ret
