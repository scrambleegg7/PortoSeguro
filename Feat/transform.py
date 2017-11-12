#
import pandas as pd
import numpy as np

from multiprocessing import *

from PortoSeguro.Feat.one_hot import onehot
from PortoSeguro.Feat.med_mean import med_mean

class MulHelper(object):

    def __init__(self, cls, mtd_name):
        self.cls = cls
        self.mtd_name = mtd_name

    def __call__(self, *args, **kwargs):
        return getattr(self.cls, self.mtd_name)(*args, **kwargs)

class Transform(object):

    def __init__(self,train_df,test_df):
        self.test = False

        self.d_median, self.d_mean = med_mean(train_df)
        self.one_hot = onehot(train_df)

        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        del train_df, test_df

    def sumVar(self):
        pass

    def transform_df(self,df):
        df = pd.DataFrame(df)
        dcol = [c for c in df.columns if c not in ['id','target']]
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
        for c in dcol:
            if '_bin' not in c: #standard arithmetic
                df[c+str('_median_range')] = (df[c].values > self.d_median[c]).astype(np.int)
                df[c+str('_mean_range')] = (df[c].values > self.d_mean[c]).astype(np.int)
                #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
                #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
                #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
                #df[c+str('_exp')] = np.exp(df[c].values) - 1
        one_hot = self.one_hot
        for c in one_hot:
            if len(one_hot[c])>2 and len(one_hot[c]) < 7:
                for val in one_hot[c]:
                    df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
        return df

    def multi_transform(self,df):
        print('Init Shape: ', df.shape)
        p = Pool(cpu_count())

        df = p.map(MulHelper(self, 'transform_df'), np.array_split(df, cpu_count()) )
        #df = p.map(self.transform_df, np.array_split(df, cpu_count()))

        df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
        p.close(); p.join()
        print('After Shape: ', df.shape)
        return df

    def excecute(self):

        new_train_df = self.transform_df(self.train_df)
        new_test_df = self.transform_df(self.test_df)
        #new_train_df = self.multi_transform(self.train_df)
        #new_test_df = self.multi_transform(self.test_df)
        col = [c for c in new_train_df.columns if c not in ['id','target']]
        col = [c for c in col if not c.startswith('ps_calc_')]

        #dups = new_train_df[new_train_df.duplicated(subset=col, keep=False)]
        #print(dups.shape)
        #new_train_df = new_train_df[~(new_train_df['id'].isin(dups['id'].values))]

        X = new_train_df.drop(["id","target"],axis=1)
        X_test = new_test_df.drop(["id"],axis=1)
        y = new_train_df["target"]
        return X, y, X_test
