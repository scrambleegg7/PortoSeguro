#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import LabelBinarizer

import scipy.interpolate
import scipy.integrate

from datetime import datetime

from sklearn import model_selection

from PortoSeguro.env import setEnv
from PortoSeguro.gini import gini_xgb
from PortoSeguro.gini import gini_normalized

from PortoSeguro.DataModelClass import DataModelClass
from PortoSeguro.gini import eval_gini

from catboost import CatBoostRegressor
from catboost import CatBoostClassifier

import gc

# Oct. 30 ver 00 : nothing to modify feature -1 -> nan --> 999

MAX_ROUNDS = 650
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.05

SET_NAN = True

def readData():

    dataCls =  DataModelClass()

    train = dataCls.readTrain()
    test = dataCls.readTest()

    print("drop all calc.... ")
    col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
    train = train.drop(col_to_drop, axis=1)
    test = test.drop(col_to_drop, axis=1)

    #cat_features = [a for a in train.columns if a.endswith('cat')]

    #
    #
    #

    print(".. set NaN for -1 data .....")
    train = train.replace(-1,np.nan)
    test = test.replace(-1,np.nan)
    train.fillna(999, inplace=True)
    test.fillna(999, inplace=True)

    if SET_NAN:

        print(' catboost SET NAN field with apply function....')

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

def process2(train,test):

    #train, test = readData()
    y = train['target']
    y_valid_pred = 0.0*y
    y_test_pred = 0

    sub=test['id'].to_frame()
    sub['target']=0

    train = train.drop(["id","target"],axis=1)
    test = test.drop(["id"],axis=1)


    cat_feature_inds = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train.columns):
        num_uniques = len(train[c].unique())
        if num_uniques < cat_unique_thresh \
           and c.endswith("cat"):
            cat_feature_inds.append(i)

    print("categorical features ......", train.columns.values[cat_feature_inds])

    X = train.values
    X_test = test.values

    print(X.shape)
    print(X_test.shape)

    # Set up folds
    K = 3
    kf = model_selection.KFold(n_splits = K, random_state = 1, shuffle = True)
    np.random.seed(0)

    #
    # cat model classifier should be used....
    #
    # Set up classifier
    model = CatBoostClassifier(
        learning_rate=LEARNING_RATE,
        depth=6,
        l2_leaf_reg = 14,
        iterations = MAX_ROUNDS,
    #    verbose = True,
        loss_function='Logloss'
    )

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(' catboost kfold: {}  of  {} : '.format(i+1, K))
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        #print("Fold : %d" % i)

        fit_model = model.fit(X_train, y_train, cat_features = cat_feature_inds)

        pred = model.predict_proba(X_valid)[:,1]
        print("    Gini = %.6f " %  eval_gini(y_valid,pred) )
        y_valid_pred.iloc[test_index] = pred

        y_test_pred += fit_model.predict_proba(X_test)[:,1]

    y_test_pred /= K

    gc.collect()
    sub.head(5)

    print("")
    print("    Gini = %.6f " %  eval_gini(y,y_valid_pred) )

    sub['target'] = y_test_pred
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub.to_csv('output/model31_catb_{}.csv.gz'.format(d), index=False, float_format='%.5f',compression="gzip")

def main():

    #y = np.random.randn(100)
    #y_pred = np.random.randn(100)

    #print(eval_gini(y,y_pred))
    train, test = readData()
    process2(train,test)



if __name__ == "__main__":
    main()
