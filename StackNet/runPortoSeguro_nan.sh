#!/bin/sh

#
# 201610
#
DATADIR=/Users/donchan/Documents/myData/KaggleData/PortoSeguro/output
TRAIN=$DATADIR/dataset2_train_20171021_185538_nan.txt
TEST=$DATADIR/dataset2_test_20171021_185648_nan.txt
PREDS=$DATADIR/stack_PortoSeguro_pred.csv

CONFIGDIR=/Users/donchan/Documents/Statistical_Mechanics/tensorflow/PortoSeguro/StackNet
#PARAMS=$CONFIGDIR/tune_single.txt
PARAMS=$CONFIGDIR/dataset2_params_level2.txt

JARDIR=/Users/donchan/SourceCodes/StackNet

/usr/bin/java -Xmx12048m -jar $JARDIR/StackNet.jar train task=classification \
    sparse=true has_head=false output_name=zillow_lv2_201610 model=zillow_model \
    pred_file=$PREDS train_file=$TRAIN test_file=$TEST test_target=false \
    params=$PARAMS verbose=true threads=6 metric=accuracy \
    stackdata=false seed=1 folds=4 Bins=3
