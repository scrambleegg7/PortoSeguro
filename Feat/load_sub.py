
"""
__file__
    load_test.py
__description__
    This file provides test case of subroutine for Feature engineering.
__author__
    Hideaki Hamano
"""

from PortoSeguro.DataModelClass import DataModelClass

def load_sub():

    dataCls = DataModelClass()
    sub_df = dataCls.readSampleSub()

    return sub_df
