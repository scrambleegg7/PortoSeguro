#
import pandas as pd
import numpy as np

def replace_nan(_df):
    df = _df.copy()
    print("replace -1 with nan....")
    df = df.replace(-1, np.nan)
    print("replaced done...")
    del _df
    return df
