#
# zillow Data path
#

import numpy as np
import os

from sys import platform

from os.path import join

def setEnv():

    envs = {}
    if platform == "linux":
        envs["data_dir"] = "/home/donchan/Documents/myData/KaggleData/PortoSeguro"
    else:
        envs["data_dir"] = "/Users/donchan/Documents/myData/KaggleData/PortoSeguro"

    envs["train"] = join(envs["data_dir"],"train.csv")
    envs["test"] = join(envs["data_dir"],"test.csv")

    envs["sample_submission"] = join(envs["data_dir"],"sample_submission.csv")


    return envs
