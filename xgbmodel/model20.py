#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import seaborn as sns
import scipy.interpolate
import scipy.integrate

from datetime import datetime

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
from sklearn import model_selection

from PortoSeguro.env import setEnv
from PortoSeguro.gini import gini_xgb
from PortoSeguro.gini import gini_lgb

from PortoSeguro.DataModelClass import DataModelClass

import gc

# ver 00 0.282
# ver 01
#    kfold = 3  # need to change to 5
#
#    train.fillna(-1, inplace=True)
#    test.fillna(-1, inplace=True)
#
#


def readData():

    dataCls =  DataModelClass()

    train = dataCls.readTrain()
    test = dataCls.readTest()

    col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
    train = train.drop(col_to_drop, axis=1)
    test = test.drop(col_to_drop, axis=1)

    #
    #
    #
    print(".. set NaN for -1 data .....")
    temp_train = train.copy()
    train = train.replace(-1,np.nan)

    temp_test = test.copy()
    test = test.replace(-1,np.nan)

    for c in train.columns.tolist():
        if c in ["id","target"]:
            continue
        else:
            train[c + "__nan__"] = train[c].apply(lambda x:1 if pd.isnull(x) else 0)

    for c in test.columns.tolist():
        if c in ["id"]:
            continue
        else:
            test[c + "__nan__"] = test[c].apply(lambda x:1 if pd.isnull(x) else 0)

    #
    # change data types
    #
    print("  change data types train ... float32 and int8 ...")
    for c in train.select_dtypes(include=['float64']).columns:
        train[c]=train[c].astype(np.float32)
    for c in train.select_dtypes(include=['int64']).columns[2:]:
        train[c]=train[c].astype(np.int8)

    print("  change data types test ... float32 and int8 ...")
    for c in test.select_dtypes(include=['float64']).columns:
        test[c]=test[c].astype(np.float32)
    for c in test.select_dtypes(include=['int64']).columns[2:]:
        test[c]=test[c].astype(np.int8)


    print("** train shape .. ",train.shape)
    print("** test shape .. ",test.shape)

    return train, test

def process2():

    train, test = readData()

    non_categorical = ["ps_calc_01","ps_calc_02","ps_calc_03"]
    non_categorical.extend( ["ps_car_12","ps_car_13","ps_car_14","ps_car_15"]  )
    non_categorical.extend( ["ps_reg_01","ps_reg_02","ps_reg_03"]  )

    # check missing value
    missing_perc_thresh = 0.90
    exclude_missing = []
    num_rows = train.shape[0]
    for c in train.columns:
        num_missing = train[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(num_rows)
        if missing_frac > missing_perc_thresh:
            exclude_missing6.append(c)
    print("We exclude: %s" % exclude_missing)
    print(len(exclude_missing))

    exclude_unique = []
    for c in train.columns:
        num_uniques = len(train[c].unique())
        if train[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1 and not "__nan__" in c:
            exclude_unique.append(c)
    print("**  We exclude: %s" % exclude_unique)
    print(len(exclude_unique))

    exclude_other = ['id', 'target']  # for indexing/training only
    train_features = []
    for c in train.columns:
        if c not in exclude_missing \
           and c not in exclude_other and c not in exclude_unique:
            train_features.append(c)
    print("We use these for training: %s" % train_features)
    print(len(train_features))

    cat_feature_inds = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train_features):
        num_uniques = len(train[c].unique())
        if num_uniques < cat_unique_thresh \
           and not c in non_categorical:
            cat_feature_inds.append(i)

    print("** Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

    xgb_params = {'eta': 0.02,
                 'max_depth': 6,
                 'subsample': 0.9,
                 'colsample_bytree': 0.9,
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc',
                 'seed': 99,
                 'silent': True}

    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    X = train[train_features]
    features = X.columns
    X = X.values
    y = train['target'].values
    sub=test['id'].to_frame()
    sub['target']=0

    nrounds=10000  # need to change to 2000

    # kfold = 5  # need to change to 5

    kfold = 3  # need to change to 5
    skf = model_selection.StratifiedKFold(n_splits=kfold, random_state=0)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        d_train = xgb.DMatrix(X_train, y_train)
        d_valid = xgb.DMatrix(X_valid, y_valid)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        xgb_model = xgb.train(xgb_params, d_train, nrounds, watchlist, early_stopping_rounds=500,
                              feval=gini_xgb, maximize=True, verbose_eval=200)
        sub['target'] += xgb_model.predict(xgb.DMatrix(test[features].values),
                            ntree_limit=xgb_model.best_ntree_limit) / (kfold)
    gc.collect()
    sub.head(2)
    d = datetime.now().strftime("%Y%m%d_%H%M%S")

    sub.to_csv('output/model20_xgb_{}.csv.gz'.format(d), index=False, float_format='%.5f',compression="gzip")


def main():

    process2()




if __name__ == "__main__":
    main()
