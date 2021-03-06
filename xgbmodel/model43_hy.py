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

from PortoSeguro.Feat.load_znullSum import load_data
from PortoSeguro.Feat.load_sub import load_sub


# hyperopt for tuning xgboost parameters
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials




def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]


dataCls = DataModelClass()
train_df = dataCls.readTrain()
test_df = dataCls.readTest()
sub = dataCls.readSampleSub()

X, y, test_df = smoothing(train_df,test_df)
#train_df, y, test_df = load_data()
id_train = train_df["id"]

f_cats = [f for f in X.columns if "_cat" in f]

y_valid_pred = 0*y
y_test_pred = 0


print(X.shape,y.shape,test_df.shape)

# Set up classifier

def score(params):
    print("Training with params : ")
    print(params)
    N_boost_round=[]

    # Set up folds
    K = 5
    kf = KFold(n_splits = K, random_state = 1, shuffle = True)
    np.random.seed(0)
    Score=[]
    # for getting mean value of test valuation from K fold
    # finally devide with K....
    S_test = np.zeros( (test_df.shape[0],K)  )
    #skf = cross_validation.StratifiedKFold(y_train, n_folds=5, shuffle=True,random_state=25)
    for i, (train_index, test_index) in enumerate(kf.split(train_df)):

        # Create data for this fold
        y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
        X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
        X_test = test_df.copy()

        d = datetime.now().strftime("%Y%m%d_%H%M%S")
        print( "\nFold {} Starting time: {}".format( i+1, d  )   )
        # Enocode data
        for f in f_cats:
            fldname = f + "_avg"
            train_avg, valid_avg, test_avg = target_encode(
                                        trn_series=X_train[f],
                                        val_series=X_valid[f],
                                        tst_series=X_test[f],
                                        target=y_train,
                                        min_samples_leaf=200,
                                        smoothing=10,
                                        noise_level=0
                                        )

            X_train = X_train.assign(fldname = train_avg)
            X_valid = X_valid.assign(fldname = valid_avg)
            X_test = X_test.assign(fldname = test_avg)


        model = XGBClassifier(
                                n_estimators=MAX_ROUNDS,
                                max_depth=int(params["max_depth"]),
                                objective="binary:logistic",
                                learning_rate=LEARNING_RATE,
                                subsample=params["subsample"],
                                min_child_weight=params["min_child_weight"],
                                colsample_bytree=params["colsample_bytree"],
                                scale_pos_weight=params["scale_pos_weight"],
                                gamma=params["gamma"],
                                reg_alpha=params["reg_alpha"],
                                reg_lambda=params["reg_lambda"],
                             )
        #model = XGBClassifier(params)
        # Run model for this fold
        if OPTIMIZE_ROUNDS:
            eval_set=[(X_valid,y_valid)]
            fit_model = model.fit( X_train, y_train,
                                   eval_set=eval_set,
                                   eval_metric=gini_xgb,
                                   early_stopping_rounds=EARLY_STOPPING_ROUNDS+50,
                                   verbose=False
                                 )
            print( "  Best N trees = ", model.best_ntree_limit )
            print( "  Best gini = ", model.best_score )
            #score = -1. * model.best_score
        else:
            fit_model = model.fit( X_train, y_train )

        pred = fit_model.predict_proba(X_valid)[:,1]
        y_valid_pred[test_index] = pred
        #y_valid_pred.loc[test_index] = np.log( 1. / ( 1. - pred)    )
        print( "  Best Gini = %.6f " % (eval_gini(y_valid, pred)) )

        # Accumulate test set predictions
        probability = fit_model.predict_proba(X_test)[:,1]
        S_test[:,i] = probability

        score = eval_gini(y_valid, pred)
        Score.append(score)

    y_test_pred = S_test.mean(axis=1)  # Average test set predictions
    #y_test_pred = 1. / ( 1 + np.exp( -y_test_pred ) )

    Average_best_score = np.average(Score)
    print("\tAvg. Score {0}\n\n".format(Average_best_score) )

    print("Save validation predictions for stacking/ensembling")
    val = pd.DataFrame()
    val['id'] = id_train
    val['target'] = y_valid_pred.values
    #val['target'] = 1. / ( 1. + np.exp( -y_valid_pred.values ) )

    sub['target'] = y_test_pred

    param_df = pd.DataFrame([params])
    params["Score"] = Average_best_score
    str_score = str(Average_best_score)
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_df.to_csv('output/smoothing/model43_xgb_hyperopt_params_{}_{}.csv'.format( str_score,d), index=False)
    val.to_csv('output/smoothing/model43_xgb_hyperopt_valid_{}_{}.csv'.format(str_score,d), float_format='%.6f', index=False)
    sub.to_csv('output/smoothing/model43_xgb_hyperopt_submission_{}_{}.csv'.format(str_score,d), float_format='%.6f', index=False)

    return {'loss': Average_best_score, 'status': STATUS_OK}


def optimize(trials):
    space = {
        "objective": "binary:logistic",
        "learning_rate":hp.quniform("learning_rate", 0.06, 0.08, 0.01),
        #Control complexity of model
#        "eta" : hp.quniform("eta", 0.2, 0.6, 0.05),

        "max_depth": hp.choice('max_depth', np.arange(2, 10, dtype=int)),
        #error code ....
        #"max_depth" : hp.quniform("max_depth", 1, 10, 1),
        "min_child_weight" : hp.quniform('min_child_weight', 1, 10, 1),
        'gamma' : hp.quniform('gamma', 5, 15, 0.5),

        #Improve noise robustness
        "subsample" : hp.quniform('subsample', 0.5, 1, 0.05),
        "colsample_bytree" : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        "scale_pos_weight" : hp.quniform('scale_pos_weight', 0.8, 2, 0.05),
        "reg_alpha" : hp.quniform('reg_alpha',1, 8,0.5),
        "reg_lambda" : hp.quniform('reg_lambda',1, 3,0.05),
        'silent' : 1}

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=100)
    print("best parameters",best)

trials = Trials()
optimize(trials)
