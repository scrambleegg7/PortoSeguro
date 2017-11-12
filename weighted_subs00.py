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

# load base
dataCls =  DataModelClass()

train = dataCls.readTrain()
sub = dataCls.readSampleSub()
#print(train.shape)

y_values = train["target"].values
print("mean & std.", np.mean(y_values), np.std(y_values))

xgb_output = "xgbmodel/output/znull/model42_xgb_hyperopt_params_0.28*"

files = glob(xgb_output)
for f in files:

    param_df = pd.read_csv(f)
    print(param_df.head())

#print(files)
#submission.to_csv(#
#    output,
#    float_format='%.4f',
#    index=False)
#print("Done! Good luck with submission # :)" )

print ("done")
