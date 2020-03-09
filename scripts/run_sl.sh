#!/usr/bin/env bash

BATCH_SIZE=$1
MODEL=$2 # "cvgg2gap", "cvgg2vh", "cvgg2hv", "alekseiori", "alekseivh", "alekseihv"
COMMENT=$3

./scripts/train_sl.sh 1 ${BATCH_SIZE} 50 50 ${MODEL} "sl" $COMMENT
./scripts/train_sl.sh 1 ${BATCH_SIZE} 100 100 ${MODEL} "sl" $COMMENT
./scripts/train_sl.sh 1 ${BATCH_SIZE} 500 500 ${MODEL} "sl" $COMMENT
./scripts/train_sl.sh 1 ${BATCH_SIZE} 1000 1000 ${MODEL} "sl" $COMMENT
