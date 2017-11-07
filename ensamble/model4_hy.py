# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

MAX_ROUNDS = 370
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.07
EARLY_STOPPING_ROUNDS = 50
# Note: I set EARLY_STOPPING_ROUNDS high so that (when OPTIMIZE_ROUNDS is set)
#       I will get lots of information to make my own judgment.  You should probably
#       reduce EARLY_STOPPING_ROUNDS if you want to do actual early stopping.

"""
This simple scripts demonstrates the use of xgboost eval results to get the best round
for the current fold and accross folds.
It also shows an upsampling method that limits cross-validation overfitting.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import gc
from numba import jit
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime

from PortoSeguro.DataModelClass import DataModelClass
from PortoSeguro.gini import eval_gini
from PortoSeguro.Feat.smoothing_feat import smoothing
from PortoSeguro.Feat.smoothing_feat import target_encode
from PortoSeguro.Feat.smoothing_feat import add_noise

from PortoSeguro.Feat.load_data import load_data
from PortoSeguro.Feat.load_sub import load_sub


# hyperopt for tuning xgboost parameters
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

X, y, X_test = load_data()
# Set up classifier

def score(params):
    print("Training with params : ")
    print(params)
    N_boost_round=[]

    # Set up folds
    K = 5
    #kf = KFold(n_splits = K, random_state = 1, shuffle = True)
    folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=42).split(X, y))

    np.random.seed(0)
    Score=[]
    #skf = cross_validation.StratifiedKFold(y_train, n_folds=5, shuffle=True,random_state=25)
    for i, (train_index, test_index) in enumerate(folds):
        # Create data for this fold
        y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
        X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
        print( "\nFold ", i+1)
        print(X_train.shape,y_train.shape)

        lgb_params4 = {}
        lgb_params4['n_estimators'] = int(params["n_estimators"])
        lgb_params4['max_bin'] = params["max_bin"]
        lgb_params4['max_depth'] = params["max_depth"]
        lgb_params4['learning_rate'] = 0.25 # shrinkage_rate
        lgb_params4['boosting_type'] = 'gbdt'
        lgb_params4['objective'] = 'binary'
        lgb_params4['min_data'] = params["min_data"]        # min_data_in_leaf
        lgb_params4['min_hessian'] = params["min_hessian"]     # min_sum_hessian_in_leaf
        lgb_params4['num_leaves'] = 16     # min_sum_hessian_in_leaf
        lgb_params4['verbose'] = 0

        model = LGBMClassifier(**lgb_params4)
        model.fit( X_train, y_train )

        pred = model.predict_proba(X_valid)[:,1]
        print( "  Best Gini = %.6f " % (eval_gini(y_valid, pred)) )

        score = eval_gini(y_valid, pred)
        Score.append(score)

    Average_best_score = np.average(Score)
    print("\tAvg. Score {0}\n\n".format(Average_best_score) )

    params["Score"] = Average_best_score
    param_df = pd.DataFrame([params])
    str_score = str(Average_best_score)
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_df.to_csv('output/model4_lgbm_hyperopt_params_{}_{}.csv'.format( str_score,d), index=False)

    return {'loss': Average_best_score, 'status': STATUS_OK}


def optimize(trials):
    space = {
        "objective": "binary:logistic",
        "learning_rate":hp.quniform("learning_rate", 0.02, 0.05, 0.01),
        "n_estimators":hp.quniform("n_estimators",600,1000,100),
        "max_bin": hp.choice('max_bin', np.arange(6, 10, dtype=int)),
        "max_depth": hp.choice('max_depth', np.arange(4, 6, dtype=int)),
        #Improve noise robustness
        "min_data" : hp.quniform('min_data',1, 8,0.5),
        "min_hessian" : hp.quniform('min_hessian',0.01,0.1, 0.02),
        "num_leaves" : hp.quniform('num_leaves',16,64,4)
        }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=10)
    print("best parameters",best)

trials = Trials()
optimize(trials)
