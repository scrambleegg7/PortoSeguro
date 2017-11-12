# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 01:34:03 2017

@author: mimar
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

from scipy.stats.mstats import hmean

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
base_xgb="ensamble/other_results/*" # stacknet preds
gp="gp/output/gpari.csv"
gp_df = pd.read_csv(gp)
#print("gp:",gp_df.shape)

xgb_smoothing="xgbmodel/smoothing/xgb_smoothing_weigted_0.28646140794592423_20171112214430.csv" # stacknet preds

std_total = 0
stds = []
targets = []
print("first, read predesined data  ....")
for f in glob(base_xgb):
    #print(f)
    filename = f.split('/')[-1]
    valid_df = pd.read_csv(f)
    m, s = valid_df["target"].mean(), valid_df["target"].std()
    del valid_df
    print(filename,m,s)
    stds.append(s)
    std_total += s

    df = pd.read_csv(f)
    targets.append( df["target"] )

gp_target = gp_df["target"]

preds = pd.concat([ targets[0] , targets[1], targets[2], targets[3]   ])
print("concat 4 type files and header info....")
preds = preds.groupby(level=0).apply(hmean)
preds = preds * 0.40 + gp_target * 0.71
print(preds.head())
print(preds.shape)
#score = eval_gini(y_values, preds)
#print("Harmomic mean gini from predesigned results ...",  score)

submission_df["target"] = preds

cdate = datetime.now().strftime("%Y%m%d%H%M%S")
output="ensamble/other_results/ensamble_harmonicmean_{}.csv.gz".format(cdate)# file to generate submissions to

submission_df.to_csv(
    output,
    index=False,compression="gzip")
print("Done! Good luck with submission # :)" )




# Nov. 4th new xgb model .. log odds

print("done")
