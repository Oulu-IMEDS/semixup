#!/usr/bin/env bash

FOLD=1 # start from 1
BATCH_SIZE=20
# num of labels per KL grade
N_DATA=$1
# 'semixup' 'ict' 'mixmatch', 'pimodel', 'sl'
METHOD_NAME=$2
COMMENT=$3
MODEL_NAME="cvgg2hv"

DATA_DIR="${PWD}/scripts/processed_data/MOST_OAI_00_0_2_cropped"

bash ${PWD}/scripts/exp_ssl_settings.sh ${FOLD} ${BATCH_SIZE} ${N_DATA} ${COMMENT} ${MODEL_NAME} ${METHOD_NAME} ${DATA_DIR} 'none'
