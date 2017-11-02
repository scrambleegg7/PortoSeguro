#

import numpy as np
import pandas as pd

def onehot(train_df):
    return {c: list(train_df[c].unique()) for c in train_df.columns if c not in ['id','target']}
