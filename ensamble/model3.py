#
import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from PortoSeguro.DataModelClass import DataModelClass
from PortoSeguro.EnsambleClass import EnsembleClass
#from PortoSeguro.Feat.load_data_sum_znull import load_data
from PortoSeguro.Feat.load_data import load_data

from PortoSeguro.Feat.load_sub import load_sub

train, target_train, test = load_data()
sub = load_sub()

# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.03
lgb_params['n_estimators'] = 900
#lgb_params['max_bin'] = 3
lgb_params['subsample'] = 0.9
#lgb_params['num_leaves'] = 8
#lgb_params['max_depth'] = 6
#lgb_params['max_bin'] = 3
lgb_params['lambda_l1'] = 5
#lgb_params['lambda_l1'] = 10
#lgb_params['min_sum_hessian_in_leaf'] = 900
#lgb_params['min_data_in_leaf'] = 900

#lgb_params['subsample_freq'] = 10
#lgb_params['colsample_bytree'] = 0.7
#lgb_params['min_child_samples'] = 500
#lgb_params['random_state'] = 99

lgb_params2 = {}
lgb_params2['n_estimators'] = 1000
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 5
#lgb_params2['random_state'] = 99

lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
#lgb_params3['random_state'] = 99

# XGBoost params
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.02
xgb_params['n_estimators'] = 650
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.8
xgb_params['colsample_bytree'] = 0.8
xgb_params['min_child_weight'] = .77
xgb_params['scale_pos_weight'] = 1.6
xgb_params['gamma'] = 10
xgb_params['reg_alpha']=8
xgb_params['reg_lambda']=1.3

lgb_params4 = {}
lgb_params4['n_estimators'] = 1450
lgb_params4['max_bin'] = 20
lgb_params4['max_depth'] = 6
lgb_params4['learning_rate'] = 0.25 # shrinkage_rate
lgb_params4['boosting_type'] = 'gbdt'
lgb_params4['objective'] = 'binary'
lgb_params4['min_data'] = 500         # min_data_in_leaf
lgb_params4['min_hessian'] = 0.05     # min_sum_hessian_in_leaf
lgb_params4['verbose'] = 0


lgb_model = LGBMClassifier(**lgb_params)
lgb_model2 = LGBMClassifier(**lgb_params2)
lgb_model3 = LGBMClassifier(**lgb_params3)
lgb_model4 = LGBMClassifier(**lgb_params3)

xgb_model = XGBClassifier(**xgb_params)

log_model = LogisticRegression()
stack = EnsembleClass(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model,lgb_model2))


#        base_models = (lgb_model,lgb_model2,lgb_model3,lgb_model4,xgb_model))

y_pred = stack.fit_log_predict(train, target_train, test)

#############################################

#Stacker score: 0.6404 AUC, LB: 0.281 Gini
#Stacker score: 0.6420 AUC, LB: 0.282 Gini
#Stacker score: 0.64218 AUC, LB: 0.283 Gini
#Stacker score: 0.64243 AUC, LB: 0.283 Gini
#Stacker score: 0.64268 AUC, LB: 0.284 Gini

#############################################


#sub = pd.DataFrame()
#sub['id'] = id_test
sub['target'] = y_pred


d = datetime.now().strftime("%Y%m%d_%H%M%S")
sub.to_csv('output/stacked_3_{}.csv'.format(d), index=False)
