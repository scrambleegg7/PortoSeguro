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
from PortoSeguro.Feat.transform import Transform

#from sumVar import setFeaturesSumAndVar
#from znull import zNullCount

def load_data():

    dataCls = DataModelClass()

    train_df = dataCls.readTrain()
    test_df = dataCls.readTest()

    tr = Transform(train_df, test_df)
    X,y,X_test = tr.excecute()
    print(X.shape,y.shape,X_test.shape)

    return X,y,X_test

def main():
    load_data()

if __name__ == "__main__":
    main()
