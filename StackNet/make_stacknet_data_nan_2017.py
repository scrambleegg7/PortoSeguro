# Based on this kaggle script : https://www.kaggle.com/danieleewww/xgboost-lightgbm-and-olsv107-w-month-features/code

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from PortoSeguro.env import setEnv
from PortoSeguro.gini import gini_xgb
from PortoSeguro.gini import gini_lgb
from PortoSeguro.gini import gini_normalized

from PortoSeguro.DataModelClass import DataModelClass

from datetime import datetime

directory="/home/donchan/Documents/myData/KaggleData/PortoSeguro" # hodls the data
out_dir="/home/donchan/Documents/myData/KaggleData/PortoSeguro/output" # hodls the data

def setFeaturesSumAndVar(_df):

    properties=_df.copy()
    print("\n")
    print("-"*30)
    print("\n    set Sum/Var of Features of properties ...")
    print("-"*30)
    properties["features_sum"] = properties.sum(axis=1).values.reshape(-1,1)
    #properties["features_var"] = properties.var(axis=1).values.reshape(-1,1)

    scaler = StandardScaler()
    target_list = ["features_sum","features_var"]
    #for c in target_list:
    properties["features_sum"] = scaler.fit_transform(properties["features_sum"].values.reshape(-1,1))

    del _df
    return properties

## converts arrayo to sparse svmlight format
def fromsparsetofile(filename, array, deli1=" ", deli2=":",ytarget=None):
    zsparse=csr_matrix(array)
    indptr = zsparse.indptr
    indices = zsparse.indices
    data = zsparse.data
    print(" data lenth %d" % (len(data)))
    print(" indices lenth %d" % (len(indices)))
    print(" indptr lenth %d" % (len(indptr)))

    ofilename = os.path.join(out_dir,filename)
    f=open(ofilename,"w")

    counter_row=0
    for b in range(0,len(indptr)-1):
        #if there is a target, print it else , print nothing
        if len(ytarget) > 0:
             f.write(str(ytarget[b]) + deli1)

        for k in range(indptr[b],indptr[b+1]):
            if (k==indptr[b]):
                if np.isnan(data[k]):
                    f.write("%d%s%f" % (indices[k],deli2,-1))
                else :
                    f.write("%d%s%f" % (indices[k],deli2,data[k]))
            else :
                if np.isnan(data[k]):
                     f.write("%s%d%s%f" % (deli1,indices[k],deli2,-1))
                else :
                    f.write("%s%d%s%f" % (deli1,indices[k],deli2,data[k]))
        f.write("\n")
        counter_row+=1
        if counter_row%10000==0:
            print(" row : %d " % (counter_row))
    f.close()

#creates the main dataset abd prints 2 files to dataset2_train.txt and  dataset2_test.txt
def changeDataTypes(_df):
    #
    # change data type from float64 to float32
    #
    properties = _df.copy()
    print("\n")
    print("-"*40)
    print( "\n** change data types float64 --> float32 ...")
    for col in properties.columns:
        if properties[col].dtype != object:
            if properties[col].dtype == float:
                properties[col] = properties[col].astype(np.float32)

    #print(properties["taxamount"].describe().transpose() )

                #print("%s %s" % (col, properties[col].dtype))
    del _df
    return properties


def readData():

    dataCls =  DataModelClass()

    train = dataCls.readTrain()
    test = dataCls.readTest()

    col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
    train = train.drop(col_to_drop, axis=1)
    test = test.drop(col_to_drop, axis=1)

    #
    #
    #
    print(".. set NaN for -1 data .....")
    temp_train = train.copy()
    train = train.replace(-1,np.nan)

    temp_test = test.copy()
    test = test.replace(-1,np.nan)

    for c in train.columns.tolist():
        if c in ["id","target"]:
            continue
        else:
            train[c + "__nan__"] = train[c].apply(lambda x:1 if pd.isnull(x) else 0)

    for c in test.columns.tolist():
        if c in ["id"]:
            continue
        else:
            test[c + "__nan__"] = test[c].apply(lambda x:1 if pd.isnull(x) else 0)

    numberOfNullCols = train.isnull().sum(axis=1)
    numberOfNullCols_test = test.isnull().sum(axis=1)

    train["znull"] = numberOfNullCols
    test["znull"] = numberOfNullCols_test
    #
    # change data types
    #
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


    print("** train shape .. ",train.shape)
    print("** test shape .. ",test.shape)

    return train, test

def dataset2():

    ##### RE-READ PROPERTIES FILE

    train, test = readData()

    non_categorical = ["ps_calc_01","ps_calc_02","ps_calc_03"]
    non_categorical.extend( ["ps_car_12","ps_car_13","ps_car_14","ps_car_15"]  )
    non_categorical.extend( ["ps_reg_01","ps_reg_02","ps_reg_03"]  )

    # check missing value
    missing_perc_thresh = 0.90
    exclude_missing = []
    num_rows = train.shape[0]
    for c in train.columns:
        num_missing = train[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(num_rows)
        if missing_frac > missing_perc_thresh:
            exclude_missing6.append(c)
    print("We exclude: %s" % exclude_missing)
    print(len(exclude_missing))

    exclude_unique = []
    for c in train.columns:
        num_uniques = len(train[c].unique())
        if train[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1 and not "__nan__" in c:
            exclude_unique.append(c)
    print("**  We exclude: %s" % exclude_unique)
    print(len(exclude_unique))

    exclude_other = ['id', 'target']  # for indexing/training only
    train_features = []
    for c in train.columns:
        if c not in exclude_missing \
           and c not in exclude_other and c not in exclude_unique:
            train_features.append(c)
    print("We use these for training: %s" % train_features)
    print(len(train_features))

    cat_feature_inds = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train_features):
        num_uniques = len(train[c].unique())
        if num_uniques < cat_unique_thresh \
           and not c in non_categorical:
            cat_feature_inds.append(i)

    print("** Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

    train.fillna(-999, inplace=True)
    test.fillna(-999, inplace=True)

    x_train = train[train_features]
    x_test = test[x_train.columns]
    y_train = train["target"].values.astype(np.float32)
    x_train = x_train.values.astype(np.float32, copy=False)
    x_test = x_test.values.astype(np.float32, copy=False)

    print('After removing outliers:')
    print (" shapes of dataset 2 ", x_train.shape, y_train.shape, x_test.shape)
    print("training....",x_train[:5])
    print("y.....",y_train[:5])
    print("test.....",x_test[:5])

    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainf = "dataset2_train_{}_nan.txt".format(d)
    print (" printing %s " % (trainf) )
    fromsparsetofile(trainf, x_train, deli1=" ", deli2=":",ytarget=y_train)

    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    testf = "dataset2_test_{}_nan.txt".format(d)
    print (" printing %s " % (testf) )
    emptyarr = []
    fromsparsetofile(testf, x_test, deli1=" ", deli2=":",ytarget=emptyarr)

    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    print (" finished with daatset2 {}".format(d) )

def main():


    dataset2()


    print( "\nFinished ...")




if __name__ == '__main__':
   main()
