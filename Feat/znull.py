
import numpy as np
import pandas as pd

from PortoSeguro.Feat.replace_nan import replace_nan

def zNullCount(df):
    df = replace_nan(df)
    print("zero null count.....")
    numberOfNullCols7 = df.isnull().sum(axis=1)
    df = df.assign(znull=numberOfNullCols7)
    df.fillna(-1,inplace=True)
    return df
