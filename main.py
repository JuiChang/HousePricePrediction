import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from HousePricePrediction_functions import cat_to_num
from HousePricePrediction_functions import col_null_count
from HousePricePrediction_functions import rm_sparse_row_col
from HousePricePrediction_functions import impute_mode_aver
from HousePricePrediction_functions import create_dum_vari
from HousePricePrediction_functions import binary_feature_selection_by_averYdiff
from HousePricePrediction_functions import numerical_feature_selection_by_CC
import xgboost as xgb

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
combine = combine.reset_index(drop=True)
print("combine set size : " + str(len(combine)) + '\n')

train_y = train['SalePrice'].copy()
print('#missing value in train_y : ' + str(train_y.isnull().sum()) + '\n')

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

# removing sparse rows and columns
ret = rm_sparse_row_col(train_conv, test_conv, train_y, 0.3, 0.5)
train_y_dense_row = ret[0]
train_dense = ret[1]
test_follow_dense = ret[2]

# missing value count, again.
print('#missing value in each columns in train_dense')
col_null_count(train_dense)
print('#missing value in each columns in test_follow_dense')
col_null_count(test_follow_dense)

# missing value imputation (using mode or average by now)
combine_impute = train_dense.append(test_follow_dense)
combine_impute = combine_impute.reset_index(drop=True)
impute_mode_aver(combine_impute, cate_vari + order_cate_vari, numer_vari + year_vari)

# missing value count, again.
print('#missing value in each columns in combine_impute')
col_null_count(combine_impute)

print('\n' + "#variables : ", len(combine_impute.columns))
# create dummy variables
ret = create_dum_vari(combine_impute, cate_vari)
combine_dum = ret[0]
dum_columns = ret[1]
print('\n' + "#variables : ", len(combine_dum.columns))
print("set size : ", len(combine_dum))

train_dum = combine_dum.iloc[:len(train), :]
test_dum = combine_dum.iloc[len(train):, :]
test_dum = test_dum.reset_index(drop=True)

print(list(train_dum.columns.values))
print(len(list(train_dum.columns.values)))
train_dum.to_csv('./submission/train_dum.csv', index=False)

# feature selection
ret = binary_feature_selection_by_averYdiff(train_dum, train_y_dense_row, test_dum, dum_columns, 10000)
print('\n' + "#variables : ", len(ret[0].columns))
ret = numerical_feature_selection_by_CC(ret[0], train_y_dense_row, ret[1],
                                        order_cate_vari + numer_vari + year_vari, 0.05)
print('\n' + "#variables : ", len(ret[0].columns))

# prepare for fitting
trainX = ret[0].drop(['Id'], axis=1)
trainY = train_y_dense_row.copy()
testX = ret[1].drop(['Id'], axis=1)
testID = ret[1]['Id'].copy()

# fit xgb
trainX = trainX.as_matrix()
testX = testX.as_matrix()
gbhs = xgb.XGBRegressor(max_depth=3, n_estimators=300, learning_rate=0.05)
gbm = gbhs.fit(trainX, trainY)
predtestY = gbm.predict(testX)
# Make prediction for the testing set
predtestY = pd.DataFrame(
    {
        'Id': testID.tolist(),
        'SalePrice': predtestY.tolist()
    }
)
predtestY.to_csv('./submission/submission_py_xgb.csv', index=False)
scores = cross_val_score(gbhs, trainX, trainY, cv=10)
print(scores, np.mean(scores))

# benchmark
# CV
# feature selection + engineering :
#   correlation coefficient
#   PCA
# (1)benchmark : linear regression (2)just before overfit
# xgboost, RF
# ensembling(correlation btw the results)
