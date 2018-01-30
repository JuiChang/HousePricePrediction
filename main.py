import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from cat_to_num import cat_to_num
from col_null_count import  col_null_count
import math

train = pd.read_csv('./HousePrices/train.csv')
test = pd.read_csv('./HousePrices/test.csv')

# count
print('\n' + "#indep. variables : " + str(len(train.columns)-1))
print("train set size : " + str(len(train)))
print("test set size : " + str(len(test)) + '\n')

# type
print(train.dtypes)
print('\n')

cate_vari = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
             'LandContour', 'Utilities', 'LotConfig',
             'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
             'BldgType', 'HouseStyle', 'RoofStyle',
             'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
             'Foundation', 'Heating', 'CentralAir', 'Functional',
             'GarageType', 'PavedDrive', 'MiscFeature', 'SaleType',
             'SaleCondition']

order_cate_vari = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                   'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
                   'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageFinish',
                   'GarageQual', 'GarageCond', 'PoolQC', 'Fence']

numer_vari = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
              'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
              'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
              'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
              'HalfBath', 'Bedroom', 'Kitchen', 'TotRmsAbvGrd',
              'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',
              'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
              'PoolArea', 'MiscVal']

year_vari = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold'] # MoSold : month

# combine train and test
combine = train.drop(['SalePrice'], axis=1).append(test)
# print(combine)
# print(combine.index[2000])
combine = combine.reset_index(drop=True)

# .drop and .append return new dataframe(by doc)
# print('\n' + "#indep. variables : " + str(len(train.columns)-1))
# print("train set size : " + str(len(train)))

# print(combine)
# print(combine.index[2000])

print("combine set size : " + str(len(combine)) + '\n')

# missing value count
print('#missing value in each columns in train.csv')
col_null_count(train)
print('#missing value in each columns in test.csv')
col_null_count(test)

# change ordered-category variables to numerical
combine_conv = combine.copy()
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'Po'], combine_conv['ExterQual'])
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'Po'], combine_conv['ExterCond'])
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], combine_conv['BsmtQual'])
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], combine_conv['BsmtCond'])
cat_to_num(['Gd', 'Av', 'Mn', 'No', 'NA'], combine_conv['BsmtExposure'])
cat_to_num(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], combine_conv['BsmtFinType1'])
cat_to_num(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], combine_conv['BsmtFinType2'])
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'Po'], combine_conv['HeatingQC'])
cat_to_num(['SBrkr', 'FuseA', 'Mix', 'FuseF', 'FuseP'], combine_conv['Electrical'])
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'Po'], combine_conv['KitchenQual'])
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], combine_conv['FireplaceQu'])
cat_to_num(['Fin', 'RFn', 'Unf', 'NA'], combine_conv['GarageFinish'])
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], combine_conv['GarageQual'])
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], combine_conv['GarageCond'])
cat_to_num(['Ex', 'Gd', 'TA', 'Fa', 'NA'], combine_conv['PoolQC'])
cat_to_num(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA'], combine_conv['Fence'])

train_conv = combine_conv.iloc[:len(train), :]
test_conv = combine_conv.iloc[len(train):, :]
test_conv = test_conv.reset_index(drop=True)

# missing value count (after the convert)
print('#missing value in each columns in train_conv')
col_null_count(train_conv)
print('#missing value in each columns in test_conv')
col_null_count(test_conv)

# remove sparse rows and columns
sparse_row = list()
for row in range(0, len(train)):
    if train_conv.loc[row].isnull().sum() > (len(train.columns)-1)*0.3:
        sparse_row.append(row)
train_dense_row = train_conv.drop(train_conv.index[sparse_row])

dense_column = list()
for col in range(0, len(train_dense_row.columns)-1):
    if train_dense_row.iloc[:, col].isnull().sum() < len(train)/2:
        dense_column.append(col)
train_dense = train_dense_row.iloc[:, dense_column]
test_follow_dense = test_conv.iloc[:, dense_column]

# missing value count, again.
print('#missing value in each columns in train_dense')
col_null_count(train_dense)
print('#missing value in each columns in test_follow_dense')
col_null_count(test_follow_dense)

# missing value count, again.
print('#missing value in each columns in train')
col_null_count(train)
print('#missing value in each columns in test')
col_null_count(test)

# missing value imputation (using mode or average by now)
# train_impute = train_dense
# train_impute = pd.DataFrame(train_dense)
# both two assignments above lead to changes of train_dense when train_impute is changed
# contrary to df assignment of .drop, .append return
# print(train_dense['MasVnrType'].isnull().sum())
# solution : train_impute = train_dense.copy()
combine_impute = train_dense.append(test_follow_dense)
combine_impute = combine_impute.reset_index(drop=True)
for colname in cate_vari + order_cate_vari:
    if colname in list(combine_impute.columns.values):
        if combine_impute[colname].isnull().sum() > 0:
            null_list = combine_impute[colname].isnull()
            mode = combine_impute[colname].mode().values[0]
            for index in range(0, len(combine_impute)):
                if null_list[index]:
                    combine_impute[colname].iat[index] = mode
# print('MasVnrType' in cate_vari)
# print(train_impute['MasVnrType'].isnull().sum())
# print(train_dense['MasVnrType'].isnull().sum())
# print(train['MasVnrType'].isnull().sum())

for colname in numer_vari + year_vari:
    if colname in list(combine_impute.columns.values):
        if combine_impute[colname].isnull().sum() > 0:
            null_list = combine_impute[colname].isnull()
            mean = combine_impute[colname].mean()
            for index in range(0, len(combine_impute)):
                if null_list[index]:
                    combine_impute[colname].iat[index] = mean

# missing value count, again.
# print('#missing value in each columns in train_impute')
# col_null_count(train_impute)
# print(train_impute.iloc[:10, :])

# create dummy variables
combine_dum = combine_impute.copy()
for colname in cate_vari:
    if colname in list(combine_dum.columns.values):
        combine_dum = pd.concat([combine_dum, pd.get_dummies(combine_dum[colname])], axis=1)
        combine_dum = combine_dum.drop([colname], axis=1)
print('\n' + "#variables : " + str(len(combine_dum.columns)))
print("set size : " + str(len(combine_dum)))

train_dum = combine_dum.iloc[:len(train), :]
test_dum = combine_dum.iloc[len(train):, :]
test_dum = test_dum.reset_index(drop=True)

print(train_dum.iloc[:10, :10])
print(test_dum.iloc[:10, :10])

# fit linear regression
train_x = train_dum.copy()
# train_y goto rm sparse row...

# benchmark
# CV
# feature selection + engineering :
#   correlation coefficient
#   PCA
# (1)benchmark : linear regression (2)just before overfit
# xgboost, RF
# ensembling(corelation btw the results)