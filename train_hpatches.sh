#!/usr/bin/env bash

nohup python3 -utt train_hpatches.py --log > logs/log_train_hpatches.txt 2>&1 &
sleep 1s
tail -f logs/log_train_hpatches.txt

