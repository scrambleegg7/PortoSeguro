#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn import model_selection

from PortoSeguro.env import setEnv
from PortoSeguro.gini import gini_xgb
from PortoSeguro.gini import gini_lgb

from sklearn.linear_model import LogisticRegression

from PortoSeguro.DataModelClass import DataModelClass

import gc

nrounds=2000  # need to change to 2000
kfold = 3  # need to change to 5
"""
ps_ind_01     1
1       ps_car_15     1
2       ps_car_14     1
3       ps_car_13     1
4       ps_car_12     1
5   ps_car_07_cat     1
6   ps_car_06_cat     1
7   ps_car_05_cat     1
8   ps_car_04_cat     1
9   ps_car_03_cat     1
10  ps_car_02_cat     1
11  ps_car_01_cat     1
12      ps_reg_03     1
13      ps_reg_02     1
14      ps_reg_01     1
15  ps_ind_17_bin     1
16  ps_car_08_cat     1
17  ps_ind_05_cat     1
18      ps_ind_15     1
19      ps_ind_03     1
20  ps_ind_06_bin     1
21  ps_ind_07_bin     1
22  ps_ind_09_bin     1
23  ps_ind_16_bin
"""

select_features =["ps_car_15","ps_car_14","ps_car_13","ps_car_12",
                    "ps_car_07_cat","ps_car_06_cat","ps_car_05_cat","ps_car_04_cat","ps_car_03_cat","ps_car_02_cat",
                    "ps_reg_03","ps_reg_02","ps_reg_01","ps_ind_17_bin","ps_car_08_cat","ps_ind_05_cat",
                    "ps_ind_15","ps_ind_03","ps_ind_06_bin","ps_ind_07_bin","ps_ind_09_bin","ps_ind_16_bin" ]

def transform(_df):

    df = _df.copy()
    print("replace NaN...")
    df = df.replace(-1, np.NaN)

    print("counting null value by row....")
    numberOfNullCols = df.isnull().sum(axis=1)
    df["znull"] = numberOfNullCols

    print("median and mean......")
    d_median = df.median(axis=0)
    d_mean = df.mean(axis=0)

    print("set -1 for NaN ...")
    df = df.fillna(-1)
    one_hot = {c: list(df[c].unique()) for c in df.columns if c not in ['id','target']}

    dcol = [c for c in df.columns if c not in ['id','target']]

    # magic columns
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    #df['negative_one_vals'] = np.sum((df[dcol]==-999).values, axis=1)
    for c in dcol:
        if '_bin' not in c:
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)

    del _df
    return df

def readData():

    print(select_features)

    dataCls =  DataModelClass()

    train = dataCls.readTrain()
    test = dataCls.readTest()

    select_features.extend(["id"])
    test = test[select_features]

    select_features.extend(["target"])
    train = train[select_features]

    sub = dataCls.readSampleSub()

    train = transform(train)
    test = transform(test)

    print("train .....")
    #print(train.columns.tolist())
    print(train.shape)

    print("test .....")
    #print(test.columns.tolist())
    print(test.shape)

    return train, test, sub


def makeData(train, test):

    col = [c for c in train.columns if c not in ['id','target']]
    col = [c for c in col if not c.startswith('ps_calc_')]

    dups = train[train.duplicated(subset=col, keep=False)]

    train = train[~(train['id'].isin(dups['id'].values))]

    target_train = train["target"]
    train = train[col]
    test = test[col]

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print("** final shape ....")
    print(train.values.shape, test.values.shape)

    return train, target_train, test

def xgb_process(X, y, test, sub):

    sub["target"] = 0

    xgb_params = {'eta': 0.02,
                 'max_depth': 4,
                 'subsample': 0.9,
                 'colsample_bytree': 0.9,
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc',
                 'silent': True}

    #xgb_params['gpu_id'] = 0
    #xgb_params['max_bin'] = 16
    #xgb_params['tree_method'] = 'gpu_exact'


    #sub=test['id'].to_frame()
    S_train = np.zeros(  X.shape[0]    )

    sub['target']=0
    y_hat = np.zeros(sub.shape[0])

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

        y_hat += xgb_model.predict(xgb.DMatrix(test.values),
                            ntree_limit=xgb_model.best_ntree_limit) / (kfold)

        S_train[test_index] = xgb_model.predict(xgb.DMatrix(X_valid),
                            ntree_limit=xgb_model.best_ntree_limit)

    sub["target"] = y_hat

    gc.collect()
    sub.head(2)
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub.to_csv('output/model23_xgb_{}.csv.gz'.format(d), index=False, float_format='%.5f',compression="gzip")

    return S_train, y_hat


def lgb_process(X,y,test,sub):

    S_train = np.zeros(  X.shape[0]    )

    sub["target"] = 0
    # lgb
    #
    #use a small max_depth and a num_of_leaves smaller than 2**max_depth, also try bagging with a small bagging frequency
    #
    params = {
            'bagging_fraction': 0.1,
            'bagging_freq': 2,
            'subsample':0.75,
            'learning_rate': 0.01,
            'metric': 'binary_logloss',
            'metric': 'auc',
            'max_depth': 4,
            'objective': 'binary',
            'feature_fraction': 0.8,
            'colsample_bytree':0.8,
            'max_bin': 8,
            'min_child_samples':500,
            'num_rounds': 2000,
             }

    y_hat = np.zeros(sub.shape[0])

    #params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':10, 'max_bin':10,  'objective': 'binary',
    #          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10,  'min_data': 500}

    skf = model_selection.StratifiedKFold(n_splits=kfold, random_state=1)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y[train_index], y[test_index]

        lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds,
                      lgb.Dataset(X_eval, label=y_eval), verbose_eval=500,
                      feval=gini_lgb, early_stopping_rounds=100)


        y_hat += lgb_model.predict(test.values,
                            num_iteration=lgb_model.best_iteration) / (kfold)

        S_train[test_index] = lgb_model.predict(X_eval,
                            num_iteration=lgb_model.best_iteration)


    sub["target"] = y_hat
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub.to_csv('output/model23_lgb_{}.csv.gz'.format(d), index=False, float_format='%.5f',compression="gzip")
    gc.collect()
    sub.head(2)

    return S_train, y_hat

def process2():

    train, test, sub = readData()
    train, target_train, test = makeData(train, test)

    X = train.values
    y = target_train.values

    S_train = np.zeros(  (X.shape[0],2)    )
    S_test =  np.zeros(  (test.shape[0],2)     )

    S_train[:,0], S_test[:,0] = xgb_process(X,y, test, sub)
    S_train[:,1], S_test[:,1] = lgb_process(X,y, test, sub)

    print("final shape of S_train / S_test...")
    print(S_train.shape, S_test.shape)

    logit = LogisticRegression()
    logit.fit(S_train, y)
    res = logit.predict_proba(S_test)[:,1]

    sub["target"] = res
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub.to_csv('output/model23_stacker_{}.csv.gz'.format(d), index=False, float_format='%.5f',compression="gzip")
    gc.collect()
    sub.head(2)


def main():

    process2()




if __name__ == "__main__":
    main()
