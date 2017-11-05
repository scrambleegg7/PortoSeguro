

#
import numpy as np
import pandas as pd
#

def recon(reg):
    integer = int(np.round((40*reg)**2))
    for a in range(32):
        if (integer - a) % 31 == 0:
            A = a
    M = (integer - A)//31
    return A, M

def split_reg03(df):

    df['ps_reg_A'] = df['ps_reg_03'].apply(lambda x: recon(x)[0])
    df['ps_reg_M'] = df['ps_reg_03'].apply(lambda x: recon(x)[1])
    df['ps_reg_A'].replace(19,-1, inplace=True)
    df['ps_reg_M'].replace(51,-1, inplace=True)

    return df
