# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 01:34:03 2017

@author: mimar
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

from PortoSeguro.DataModelClass import DataModelClass

from scipy.optimize import minimize
from PortoSeguro.gini import eval_gini
from glob import glob

dataCls =  DataModelClass()
train = dataCls.readTrain()
submission_df = dataCls.readSampleSub()
submission_df["target"] = 0.0

y_values = train["target"].values
y_m , y_s = np.mean(y_values), np.std(y_values)
print("base mean std", y_m, y_s)
del train
# load base
#print(train.shape)
base_xgb="xgbmodel/output/smoothing/model43_xgb_hyperopt_valid_0.285*" # stacknet preds
#base_xgb="xgbmodel/output/smoothing/model43_xgb_hyperopt_valid_0.28*" # stacknet preds

std_total = 0
stds = []
for f in glob(base_xgb):
    #print(f)
    params = f.replace("valid","params")
    subm = f.replace("valid","submission")
    valid_df = pd.read_csv(f)
    m, s = valid_df["target"].mean(), valid_df["target"].std()
    del valid_df
    print(m,s)
    stds.append(s)
    std_total += s

valid_target = np.zeros_like(y_values)
for i,f in enumerate(glob(base_xgb)):

    valid_df = pd.read_csv(f)
    w = stds[i] / np.sum(stds)
    print("calculated weights based on std ....", w)
    valid_target += w * valid_df["target"]
    #submission_df +=  w * test_df["target"]
    del valid_df

score = eval_gini(y_values, valid_target)
print("weighted gini...",  score)

y_test = np.zeros( submission_df.shape[0] )
for i,f in enumerate(glob(base_xgb)):
    subm = f.replace("valid","submission")
    test_df = pd.read_csv(subm)
    w = stds[i] / np.sum(stds)
    y_test += w * test_df["target"]

    del test_df

submission_df["target"] = y_test

cdate = datetime.now().strftime("%Y%m%d%H%M%S")
output="xgbmodel/output/xgb_smoothing_weigted_{}_{}.csv".format(str(score), cdate)# file to generate submissions to

submission_df.to_csv( output, index=False )
    #index=False,compression="gzip")
print("Done! Good luck with submission # :)" )




# Nov. 4th new xgb model .. log odds

print("done")
