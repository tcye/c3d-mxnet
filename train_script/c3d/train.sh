#! /bin/sh
path=$(dirname $(readlink -f "$0"))
python3 ${path}/../train_model.py ${path}\
  2>&1 | tee -a ${path}/train.log
