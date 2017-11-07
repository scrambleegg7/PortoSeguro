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

# load base
dataCls =  DataModelClass()

train = dataCls.readTrain()
sub = dataCls.readSampleSub()
#print(train.shape)

y_values = train["target"].values

# Nov. 4th new xgb model .. log odds
base_xgb="xgbmodel/output/model41_xgb_submit_20171104_094529.csv" # stacknet preds
#catb = "catboost/output/model31_catb_20171031_184210.csv"
catb = "catboost/output/model31_catb_20171105_232459.csv"
gp="gp/output/gpari.csv"


xgb_valid_oob = "xgbmodel/output/model41_xgb_valid.csv"
catb_valid_oob = "catboost/output/model31_catb_valid.csv"
gp_valid_oob = "gp/output/gpari_valid.csv"

xgb_valid_oob = pd.read_csv(xgb_valid_oob)
catb_valid_oob = pd.read_csv(catb_valid_oob)
gp_valid_oob = pd.read_csv(gp_valid_oob)


xgb_df = pd.read_csv(base_xgb)
catb_df = pd.read_csv(catb)
gp_df = pd.read_csv(gp)


predictions = np.zeros( (train.shape[0],3) )
print(predictions.shape)
predictions[:,0] = xgb_valid_oob["target"].values
predictions[:,1] = catb_valid_oob["target"].values
predictions[:,2] = gp_valid_oob["target"].values

def mae_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for idx, weight in enumerate(weights):
        final_prediction += weight * predictions[:,idx]

    return -eval_gini(y_values, final_prediction)

lls = []
wghts = []

for i in range(100):
    starting_values = np.random.uniform( size=predictions.shape[1] )
    if i % 10 == 0:
        print("loop count:",i)
    #print(starting_values.shape)
    cons = ({'type':'ineq','fun':lambda w: 1.-sum(w)})
    bounds = [(0,1)]*predictions.shape[1]

    res = minimize(mae_func, starting_values, method='COBYLA', constraints=cons,
               bounds=bounds, options={'disp': False, 'maxiter': 10000})

    lls.append(res['fun'])
    wghts.append(res['x'])

bestSC = np.min(lls)
bestWght = wghts[ np.argmin(lls)   ]

print(bestSC)
print(bestWght)
print("sum", np.sum(bestWght))


#sub["target"].values = 0.0
sub["target"] = xgb_df["target"].values * bestWght[0] + catb_df["target"].values * bestWght[1] + \
                gp_df["target"].values * bestWght[2]

print(sub.head())

cdate = datetime.now().strftime("%Y%m%d%H%M%S")
output="merged/harmonic_{}.csv.gz".format(cdate)# file to generate submissions to

sub.to_csv(
    output,
    index=False,compression="gzip")
print("Done! Good luck with submission # :)" )

#submission.to_csv(#
#    output,
#    float_format='%.4f',
#    index=False)
#print("Done! Good luck with submission # :)" )

print ("done")
