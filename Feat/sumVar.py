#

import numpy as np
import pandas as pd

from sklearn import preprocessing

def setFeaturesSumAndVar(df):

    print("\n")
    print("-"*30)
    print("\n    set Sum/Var of Features of data ...")
    print("-"*30)
    df["features_sum"] = df.sum(axis=1).values.reshape(-1,1)
    df["features_var"] = df.var(axis=1).values.reshape(-1,1)

    scaler = preprocessing.StandardScaler()
    target_list = ["features_sum","features_var"]
    for c in target_list:
        df[c] = scaler.fit_transform(df[c].values.reshape(-1,1))

    return df
