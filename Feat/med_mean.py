import numpy as np
import pandas as pd
from PortoSeguro.Feat.replace_nan import replace_nan

def med_mean(train):

    train = replace_nan(train)

    print("median mean ...")
    d_median = train.median(axis=0)
    d_mean = train.mean(axis=0)

    train.fillna(-1,inplace=True)
    print("done....")
    return d_median,d_mean
