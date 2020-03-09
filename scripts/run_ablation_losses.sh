#!/usr/bin/env bash

FOLD=1 # start from 1
BATCH_SIZE=20
# num of labels per KL grade
N_DATA=$1

METHOD_NAME=semixup

MODEL_NAME="cvgg2hv"

DATA_DIR="${PWD}/scripts/processed_data/MOST_OAI_00_0_2_cropped"

for N_DATA in 100 500
do
  echo "Num of labeled data: ${N_DATA}"
  for REMOVED_LOSS in "1" "2" "3" "none"
  do
    if [ $REMOVED_LOSS == "none" ]
    then
      COMMENT="all"
    else
      COMMENT="noloss${REMOVED_LOSS}"
    fi

    bash ${PWD}/scripts/exp_ssl_settings.sh ${FOLD} ${BATCH_SIZE} ${N_DATA} ${COMMENT} ${MODEL_NAME} ${METHOD_NAME} ${DATA_DIR} ${REMOVED_LOSS}
  done
done
