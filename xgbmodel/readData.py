#
import pandas as pd
import numpy as np

from PortoSeguro.env import setEnv

def readData():

    env = setEnv()

    train_data = env["train"]
    test_data = env["test"]

    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)

    print("train shape...",train_df.shape)
    print("test shape...",test_df.shape)

def main():

    readData()


if __name__ == "__main__":
    main()
