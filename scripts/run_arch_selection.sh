#!/usr/bin/env bash

BATCH_SIZE=$1

for MODEL in "cvgg2gap" "cvgg2vh" "cvgg2hv" "alekseiori" "alekseivh" "alekseihv"
do
  echo "Arch ${MODEL}"
  ./scripts/train_sl.sh 1 ${BATCH_SIZE} 50 50 ${MODEL} "sl" "archselection"
  ./scripts/train_sl.sh 1 ${BATCH_SIZE} 100 100 ${MODEL} "sl" "archselection"
  ./scripts/train_sl.sh 1 ${BATCH_SIZE} 500 500 ${MODEL} "sl" "archselection"
  ./scripts/train_sl.sh 1 ${BATCH_SIZE} 1000 1000 ${MODEL} "sl" "archselection"
done