# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 01:34:03 2017

@author: mimar
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

datadir="/Users/donchan/Documents/myData/KaggleData/PortoSeguro"
sample = os.path.join(datadir,"sample_submission.csv")
sub = pd.read_csv(sample, low_memory=False)

# load base

base_lgb="output/model22_lgb_20171023_111920.csv" # stacknet preds
base_xgb="output/model22_xgb_20171023_121947.csv" # stacknet preds

base_xgb_df = pd.read_csv(base_xgb)
print(base_xgb_df.shape)
base_lgb_df = pd.read_csv(base_lgb)
print(base_lgb_df.shape)

#
# take advise to use harmonic mean...
#
sub["target"] = 2. / ( 1. / base_xgb_df["target"].values + 1. / base_lgb_df["target"].values )
print(sub.head())

cdate = datetime.now().strftime("%Y%m%d%H%M%S")
output="output/xgb_lgb_harmonic_{}.csv.gz".format(cdate)# file to generate submissions to
#output=os.path.join(datadir,output)

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
