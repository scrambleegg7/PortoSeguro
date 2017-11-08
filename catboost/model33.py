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

from sklearn.ensemble import BaggingClassifier

from PortoSeguro.Feat.load_znullSum import load_data

import gc

# Oct. 30 ver 00 : nothing to modify feature -1 -> nan --> 999

MAX_ROUNDS = 850
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.05

SET_NAN = True

def readData():

    train, y, test = load_data()
    return train, y, test

def subsample_model(model,X_train,y_train,cat_feature_inds,ratio=0.10):

    t_size = X_train.shape[0]
    topN = int( t_size * ratio )
    rnd = np.random.permutation(t_size)[:topN]

    fit_model = model.fit(X_train[rnd], y_train[rnd], cat_features = cat_feature_inds)

    return fit_model

def process2(train,y,test):

    #train, test = readData()
    #y = train['target']
    y_valid_pred = 0.0*y
    y_test_pred = 0

    id_train = train['id'].values

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
    K = 5
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
    #model = BaggingClassifier(model_)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(' catboost kfold: {}  of  {} : '.format(i+1, K))
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        #print("Fold : %d" % i)


        for size in [1,10,100]: # loop by average counter ....
            print("* loop size : %d" % (size) )
            preds = 0.
            for i in range(size):
                model = subsample_model(model,X_train,y_train,cat_feature_inds,0.1)
                pred = model.predict_proba(X_valid)[:,1]
                y_valid_pred.iloc[test_index] += pred
                preds += pred

            y_valid_pred /= size
            preds /= size
            print("    Gini = %.6f " %  eval_gini(y_valid,preds) )



def main():

    #y = np.random.randn(100)
    #y_pred = np.random.randn(100)

    #print(eval_gini(y,y_pred))
    train,y, test = readData()
    process2(train,y,test)



if __name__ == "__main__":
    main()
