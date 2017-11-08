"""
__file__
    load_test.py
__description__
    This file provides test case of subroutine for Feature engineering.
__author__
    Hideaki Hamano
"""

import pandas as pd
import numpy as np

from PortoSeguro.DataModelClass import DataModelClass
#from PortoSeguro.Feat.transform import Transform

from PortoSeguro.Feat.sumVar import setFeaturesSumAndVar
from PortoSeguro.Feat.znull import zNullCount
from PortoSeguro.Feat.replace_nan import replace_nan

def load_data():

    dataCls = DataModelClass()
    train_df = dataCls.readTrain()
    test_df = dataCls.readTest()

    print("drop all calc.... ")
    col_to_drop = train_df.columns[train_df.columns.str.startswith('ps_calc_')]
    train_df = train_df.drop(col_to_drop, axis=1)
    test_df = test_df.drop(col_to_drop, axis=1)

    train_df = replace_nan(train_df)
    test_df = replace_nan(test_df)

    train_df = setFeaturesSumAndVar(train_df)
    test_df = setFeaturesSumAndVar(test_df)

    train = zNullCount(train_df)
    test = zNullCount(test_df)

    y = train_df["target"]

    print("  change data types train ... float32 and int8 ...")
    for c in train.select_dtypes(include=['float64']).columns:
        train[c]=train[c].astype(np.float32)
    for c in train.select_dtypes(include=['int64']).columns[2:]:
        train[c]=train[c].astype(np.int8)

    print("  change data types test ... float32 and int8 ...")
    for c in test.select_dtypes(include=['float64']).columns:
        test[c]=test[c].astype(np.float32)
    for c in test.select_dtypes(include=['int64']).columns[2:]:
        test[c]=test[c].astype(np.int8)

    print(train.shape,y.shape,test.shape)

    train.fillna(-999,inplace=True)
    test.fillna(-999,inplace=True)

    return train,y,test

def main():
    load_data()

if __name__ == "__main__":
    main()
