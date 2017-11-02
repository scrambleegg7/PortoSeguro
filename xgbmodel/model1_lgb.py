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



def readData():

    dataCls =  DataModelClass()

    train = dataCls.readTrain()
    test = dataCls.readTest()

    col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
    train = train.drop(col_to_drop, axis=1)
    test = test.drop(col_to_drop, axis=1)


    #
    # change data types
    #
    print("  change data types ... float32 and int8 ...")
    for c in train.select_dtypes(include=['float64']).columns:
        train[c]=train[c].astype(np.float32)
        test[c]=test[c].astype(np.float32)
    for c in train.select_dtypes(include=['int64']).columns[2:]:
        train[c]=train[c].astype(np.int8)
        test[c]=test[c].astype(np.int8)

    print("** train shape .. ",train.shape)
    print("** test shape .. ",test.shape)

    return train, test

def process2():

    train, test = readData()

    xgb_params = {'eta': 0.01,
                 'max_depth': 4,
                 'subsample': 0.9,
                 'colsample_bytree': 0.9,
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc',
                 'seed': 99,
                 'silent': True}

    #xgb_params['gpu_id'] = 0
    #xgb_params['max_bin'] = 16
    #xgb_params['tree_method'] = 'gpu_exact'
    #xgb_params['updater'] = 'grow_gpu'
    #xgb_params['gpu_id'] = 0
    #xgb_params['max_bin'] = 16
    #xgb_params['tree_method'] = 'gpu_hist'



    X = train.drop(['id', 'target'], axis=1)
    features = X.columns
    X = X.values
    y = train['target'].values
    sub=test['id'].to_frame()
    sub['target']=0

    nrounds=2000  # need to change to 2000
    kfold = 5  # need to change to 5
    skf = model_selection.StratifiedKFold(n_splits=kfold, random_state=0)

    # lgb
    #
    #use a small max_depth and a num_of_leaves smaller than 2**max_depth, also try bagging with a small bagging frequency
    #
    params = {
            'bagging_fraction': 0.1,
            'bagging_freq': 2,
            'learning_rate': 0.03,
            'metric': 'binary_logloss',
            'metric': 'auc',
            'max_depth': 9,
            'objective': 'binary',
            'num_leaves': 15,
            'bagging_seed': 1,
            'feature_fraction': 0.8,
            'feature_fraction_seed': 1,
            'max_bin': 10,
            'num_rounds': 2000,
             }

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
        sub['target'] += lgb_model.predict(test[features].values,
                            num_iteration=lgb_model.best_iteration) / (kfold)

    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub.to_csv('output/model1_lgb_{}.csv.gz'.format(d), index=False, float_format='%.5f',compression="gzip")
    gc.collect()
    sub.head(2)

def main():

    process2()




if __name__ == "__main__":
    main()
