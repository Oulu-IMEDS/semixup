#!/bin/bash

FOLD=$1
BATCH_SIZE=$2
N_DATA=$3
COMMENT=$4
MODEL_NAME=$5
METHOD_NAME=$6
DATA_DIR=$7
REMOVED_LOSSES=$8 # for Semixup only. Value: 1, 2, or 3. String not including them 3 means to use all regularizers.

let T1=${N_DATA}
let T2=${N_DATA}*2
let T3=${N_DATA}*3
let T4=${N_DATA}*4
let T5=${N_DATA}*5
let T6=${N_DATA}*6
let T0=0

SKIP=1

let BS=${BATCH_SIZE}
let a=\(5*${N_DATA}+${BS}-1\)
let b=$a/${BS}
let N_BATCHES=${SKIP}*$b
echo "Num of batches is ${N_BATCHES} ${METHOD_NAME} ${COMMENT} ${DATA_DIR}"
echo "-------------- $b ---------- "

for N_UDATA in $T1 $T2 $T3 $T4 $T5 $T6 $T0
do
  echo "Run ./scripts/train.sh ${FOLD} ${BATCH_SIZE} ${N_DATA} ${N_UDATA} ${MODEL_NAME} 32 ${N_BATCHES} ${METHOD_NAME} ${COMMENT} ${DATA_DIR} ${REMOVED_LOSSES}"
  ./scripts/train_ssl.sh ${FOLD} ${BATCH_SIZE} ${N_DATA} ${N_UDATA} ${MODEL_NAME} 32 ${N_BATCHES} ${METHOD_NAME} ${COMMENT} ${DATA_DIR} ${REMOVED_LOSSES}
done
