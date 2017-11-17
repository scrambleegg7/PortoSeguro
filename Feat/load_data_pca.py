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

from time import time

from PortoSeguro.DataModelClass import DataModelClass

# decomposition

from sklearn.decomposition import PCA, FastICA

def load_data(train_df,test_df):

    #dataCls = DataModelClass()

    #train_df = dataCls.readTrain()
    #test_df = dataCls.readTest()

    calc_col = train_df.columns.str.startswith('ps_calc_')
    calc_col = train_df.columns[calc_col]
    train_df = train_df.drop(calc_col,axis=1)
    test_df = test_df.drop(calc_col,axis=1)

    n_comp = 5

    X_train = train_df.drop(["id","target"],axis=1)
    X_test = test_df.drop(["id"],axis=1)

    train_length = X_train.shape[0]
    test_length = X_test.shape[0]

    x_all = pd.concat([X_train,X_test])

    #PCA
    print('\nRunning PCA ...')
    pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)
    x_pca = pca.fit_transform(x_all)
    print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())

    print('Individual variance contributions:')
    for j in range(n_comp):
        print(pca.explained_variance_ratio_[j])

    start_t = time()
    #FastICA
    ica = FastICA(n_components=n_comp,random_state=42)
    ica_res = ica.fit_transform(x_all)

    for i in range(1,n_comp+1):
        x_all["pca_" + str(i)] = x_pca[:,i-1]
        x_all["ica_" + str(i)] = ica_res[:,i-1]

    print("time to consume making ica...", time() - start_t )

    print("columns ...")
    print(x_all.columns.tolist()  )

    X = x_all.head(train_length)
    X_test = x_all.tail(test_length)

    y = train_df["target"]
    print(X.shape,y.shape,X_test.shape)

    return X,y,X_test

def main():
    load_data()

if __name__ == "__main__":
    main()
