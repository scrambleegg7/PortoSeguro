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
import xgboost as xgb


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import gc
from numba import jit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier

import time
from datetime import datetime

from PortoSeguro.DataModelClass import DataModelClass
from PortoSeguro.gini import eval_gini
from PortoSeguro.Feat.load_data_pca import load_data

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def modelfit(alg, X_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=300):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            feval=gini_xgb, early_stopping_rounds=early_stopping_rounds, verbose_eval=200)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(X_train, y_train, eval_metric=gini_xgb)

    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    pred = alg.predict_proba(X_train)[:,1]
    gini_score = eval_gini(y_train, pred)

    #Print model report:
    print("\nModel Report")
    print("Gini score (full train data based ...) : %.4g" % gini_score     )
    #print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

    #feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    #print(feat_imp)
    #feat_imp.plot(kind='bar', title='Feature Importances')
    #plt.ylabel('Feature Importance Score')


dataCls = DataModelClass()

train_df = dataCls.readTrain()
sub = dataCls.readSampleSub()
test_df = dataCls.readTest()

# Process data
id_train = train_df['id'].values


X,y,test_df = load_data(train_df,test_df)


y_valid_pred = 0*y
y_test_pred = 0

# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)
np.random.seed(0)

# Set up classifier
model = XGBClassifier(
                        n_estimators=MAX_ROUNDS,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=LEARNING_RATE,
                        subsample=.8,
                        min_child_weight=.77, #Oct. 30th
#                        min_child_weight=.6,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                     )

#model = XGBClassifier(
#                         learning_rate =0.1,
#                         n_estimators=1000,
#                         max_depth=5,
#                         min_child_weight=1,
#                         gamma=0,
#                         subsample=0.8,
#                         colsample_bytree=0.8,
#                         objective= 'binary:logistic',
#                         nthread=4,
#                         scale_pos_weight=1,
#                         seed=27
#                        )

modelfit(model, X.values, y.values, useTrainCV=True, cv_folds=5, early_stopping_rounds=300)

exit()

for i, (train_index, test_index) in enumerate(kf.split(train_df)):

    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    X_test = test_df.copy()
    print( "\nFold ", i)

    # Run model for this fold
    if OPTIMIZE_ROUNDS:
        eval_set=[(X_valid,y_valid)]
        fit_model = model.fit( X_train, y_train,
                               eval_set=eval_set,
                               eval_metric=gini_xgb,
                               early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                               verbose=False
                             )
        print( "  Best N trees = ", model.best_ntree_limit )
        print( "  Best gini = ", model.best_score )
    else:
        fit_model = model.fit( X_train, y_train )

    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid)[:,1]
    gini_score = eval_gini(y_valid, pred)
    print( "  Gini = %.6f " % gini_score )
    y_valid_pred.iloc[test_index] = pred

    # Accumulate test set predictions
    probability = fit_model.predict_proba(X_test)[:,1]
    #y_test_pred += np.log(1. / ( 1. - probability))
    y_test_pred += probability

    sub['target'] = probability
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub.to_csv('output/model45_xgb_test_validate_{}_{}_{}.csv'.format(i,str(gini_score),  d), float_format='%.6f', index=False)

    del X_test, X_train, X_valid, y_train

y_test_pred /= K  # Average test set predictions
#y_test_pred = 1. / ( 1 + np.exp( -y_test_pred ) )

print( "\nGini for full training set:", eval_gini(y, y_valid_pred) )

print("Save validation predictions for stacking/ensembling")
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('output/model45_xgb_valid.csv', float_format='%.6f', index=False)

print("Create submission file averaged by Kfold ....")
#sub = pd.DataFrame()
#sub['id'] = id_test
sub['target'] = y_test_pred
d = datetime.now().strftime("%Y%m%d_%H%M%S")
sub.to_csv('output/model45_xgb_submit_{}.csv'.format(d), float_format='%.6f', index=False)
