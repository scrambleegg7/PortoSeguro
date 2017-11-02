
import numpy as np
import pandas as pd

from replace_nan import replace_nan

def zNullCount(df):
    df = replace_nan(df)
    numberOfNullCols7 = df.isnull().sum(axis=1)
    df = df.assign(znull=numberOfNullCols7)
    df.fillna(-1,inplace=True)
    return df
