#!/usr/bin/env bash
if [ $# -lt 7 ]
  then
    echo "Not enough arguments supplied!"
    exit
fi

FOLD_ID=$1 # start from 1
BS=$2
LABELS=$3
UNLABELS=$4
MODEL=$5
METHOD=$6
COMMENT=$7

DATA_DIR="${PWD}/scripts/processed_data/MOST_OAI_00_0_2_cropped"

PKL_FILE=${PWD}/scripts/processed_data/Metadata/cv_split_5fold_l_${LABELS}_u_${UNLABELS}_False_col_None.pkl
cd ${PWD}/sl

LR=1e-4

python train.py --bs $BS --comment _${METHOD}_${COMMENT}_dlr_${LR}_${MODEL}_ndf${NDF}_data_${LABELS}_${UNLABELS}_alekseitransform_samplingrandom_fold${FOLD_ID} \
    --kfold_split_file ${PKL_FILE} --n_epochs 500 --model_name ${MODEL} --root ${DATA_DIR}
