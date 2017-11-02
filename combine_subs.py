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

base_lgb="xgbmodel/output/model22_lgb_20171023_111920.csv" # stacknet preds
base_xgb="xgbmodel/output/model41_xgb_submit.csv" # stacknet preds

ensamble="ensamble/output/stacked_1_20171023_233558.csv"
ensamble2="ensamble/output/stacked_2_20171101_124642.csv"

gp="gp/output/gpari.csv"

catb = "catboost/output/model31_catb_20171031_184210.csv"

base_xgb_df = pd.read_csv(base_xgb)
print("xgb:",base_xgb_df.shape)
base_catb_df = pd.read_csv(catb)
print(base_catb_df.shape)

ensamble_df = pd.read_csv(ensamble2)
print("ensamble:",ensamble_df.shape)
gp_df = pd.read_csv(gp)
print("gp:",gp_df.shape)

#
# take advise to use harmonic mean...
#
#2. / ( 1. / base_xgb_df["target"].values + 1. / base_lgb_df["target"].values +
#sub["target"] = 2. / ( 1. / gp_df["target"].values + 1. / ensamble_df["target"].values )
#sub["target"] = 3. / ( 1. / gp_df["target"].values + 1. / ensamble_df["target"].values +
#                        1. / base_xgb_df["target"].values )

# LB 0.285 -- Oct. 30th 2017
#sub["target"] = (gp_df["target"].values * .201 +  ensamble_df["target"].values * .301 +
#                         base_xgb_df["target"].values * .5 )

sub["target"] = (gp_df["target"].values * .32 +  ensamble_df["target"].values * .521 +
                         base_xgb_df["target"].values * .25 + base_catb_df["target"] * .25 )


print(sub.head())

cdate = datetime.now().strftime("%Y%m%d%H%M%S")
output="merged/harmonic_{}.csv.gz".format(cdate)# file to generate submissions to
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
