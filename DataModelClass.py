#
import pandas as pd
import numpy as np

from PortoSeguro.env import setEnv

class DataModelClass(object):

    def __init__(self):

        env = setEnv()
        self.train_data = env["train"]
        self.test_data = env["test"]
        self.submit = env["sample_submission"]

    def readTrain(self):
        train_df = pd.read_csv(self.train_data)
        print("train shape...",train_df.shape)
        return train_df

    def readTest(self):
        test_df = pd.read_csv(self.test_data)
        print("test shape...",test_df.shape)
        return test_df

    def readSampleSub(self):
        submit_df = pd.read_csv(self.submit)
        print("submission shape...",submit_df.shape)
        return submit_df

def main():


    data =  DataModelClass()



if __name__ == "__main__":
    main()
