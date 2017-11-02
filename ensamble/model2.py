import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

# Regularized Greedy Forest
#from rgf.sklearn import RGFClassifier     # https://github.com/fukatani/rgf_python

from PortoSeguro.DataModelClass import DataModelClass
from datetime import datetime

from PortoSeguro.EnsambleClass import EnsembleClass

dataCls = DataModelClass()

#train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')

train = dataCls.readTrain()
test = dataCls.readTest()


##
## just test purposes
##
#train = train[:10000]
#test = test[:1000]

# Preprocessing (Forza Baseline)
id_test = test['id'].values

col = [c for c in train.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]

train = train.replace(-1, np.NaN)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)
one_hot = {c: list(train[c].unique()) for c in train.columns if c not in ['id','target']}

def transform(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c:
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)

    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            print("one_hot changed ..", c)
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df


train = transform(train)
test = transform(test)

col = [c for c in train.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]

dups = train[train.duplicated(subset=col, keep=False)]

train = train[~(train['id'].isin(dups['id'].values))]

target_train = train['target']
train = train[col]
test = test[col]
print(train.values.shape, test.values.shape)



# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8
lgb_params['min_child_samples'] = 500
#lgb_params['random_state'] = 99

lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
#lgb_params2['random_state'] = 99

lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
#lgb_params3['random_state'] = 99

# RandomForest params
#rf_params = {}
#rf_params['n_estimators'] = 200
#rf_params['max_depth'] = 6
#rf_params['min_samples_split'] = 70
#rf_params['min_samples_leaf'] = 30


# ExtraTrees params
#et_params = {}
#et_params['n_estimators'] = 155
#et_params['max_features'] = 0.3
#et_params['max_depth'] = 6
#et_params['min_samples_split'] = 40
#et_params['min_samples_leaf'] = 18

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
# CatBoost params
#cat_params = {}
#cat_params['iterations'] = 900
#cat_params['depth'] = 8
#cat_params['rsm'] = 0.95
#cat_params['learning_rate'] = 0.03
#cat_params['l2_leaf_reg'] = 3.5
#cat_params['border_count'] = 8
#cat_params['gradient_iterations'] = 4



lgb_model = LGBMClassifier(**lgb_params)
lgb_model2 = LGBMClassifier(**lgb_params2)
lgb_model3 = LGBMClassifier(**lgb_params3)
xgb_model = XGBClassifier(**xgb_params)

log_model = LogisticRegression()



stack = EnsembleClass(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2,lgb_model3,xgb_model))

y_pred = stack.fit_log_predict(train, target_train, test)


#############################################

#Stacker score: 0.6404 AUC, LB: 0.281 Gini
#Stacker score: 0.6420 AUC, LB: 0.282 Gini
#Stacker score: 0.64218 AUC, LB: 0.283 Gini
#Stacker score: 0.64243 AUC, LB: 0.283 Gini
#Stacker score: 0.64268 AUC, LB: 0.284 Gini

#############################################


sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred


d = datetime.now().strftime("%Y%m%d_%H%M%S")
sub.to_csv('output/stacked_2_{}.csv'.format(d), index=False)
