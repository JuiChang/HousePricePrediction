import math
import pandas as pd


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
        print('for start')
        if colname in list(df_dum.columns.values):
            tmp_df = pd.get_dummies(df_dum[colname])
            tmp_df = avoid_duplicate_label(df_dum, tmp_df)
            print(tmp_df.iloc[:10, :])
            print(1)
            tmp_df = tmp_df.reset_index(drop=True)
            df_dum = pd.concat([df_dum, tmp_df], axis=1, join_axes=[df_dum.index])
            # df_dum = pd.concat([df_dum, tmp_df], axis=1)
            print(list(tmp_df.columns.values)[0])
            print(df_dum[list(tmp_df.columns.values)[0]])
            print(2)
            df_dum.drop([colname], axis=1, inplace=True)
            print(3)
            new_columns += list(tmp_df.columns.values)
        print('for end')
    print(df_dum['$Artery'])
    ret.append(df_dum)
    ret.append(new_columns)
    print(ret[0]['$Artery'])
    return ret


def avoid_duplicate_label(df_big_in, df_small_in):
    df_big = df_big_in.copy()
    df_small = df_small_in.copy()
    for col_name in list(df_small.columns.values):
        print('col_name: ', col_name)
        tmp_col_name = col_name
        while tmp_col_name in list(df_big.columns.values):
            # df_small.rename(index=str, columns={tmp_col_name: '$'+tmp_col_name}, inplace=True)
            df_small = df_small.rename(index=str, columns={tmp_col_name: '$'+tmp_col_name})
            tmp_col_name = '$' + tmp_col_name
        print('tmp_col_name: ', tmp_col_name)
    print(df_small.iloc[:10, :])
    print('before return')
    return df_small

def binary_feature_selection_by_averYdiff(df_trainX_in, trainY_in, df_testX_in, vari_list, rm_propotion):
    ret = list()
    dict_aver_diff = dict()
    df_trainX = df_trainX_in.copy()
    df_testX = df_testX_in.copy()
    trainY = trainY_in.copy()
    for colname in list(df_trainX.columns.values):
        if colname in vari_list:
            aver_diff = 0
            # print(len(df_trainX))
            # print(colname)
            for i in range(0, len(df_trainX)):
                # print(i)
                if df_trainX[colname].iat[i] == 1:
                    aver_diff += trainY.iat[i]
                else:
                    aver_diff -= trainY.iat[i]
            dict_aver_diff[colname] = aver_diff / len(df_trainX)
    print(dict_aver_diff.values())
    return ret
