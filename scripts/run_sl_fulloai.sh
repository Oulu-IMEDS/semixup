#!/usr/bin/env bash
if [ $# -lt 2 ]
  then
    echo "Not enough arguments supplied!"
    exit
fi

FOLD_ID=1 # start from 1
BS=$1
COMMENT=$2

LABELS=31922
UNLABELS=0
MODEL="cvgg2hv"
METHOD="slfulloai"

DATA_DIR="${PWD}/scripts/processed_data/MOST_OAI_00_0_2_cropped"

PKL_FILE=${PWD}/scripts/processed_data/Metadata/cv_split_5fold_oai.pkl
cd ${PWD}/sl

LR=1e-4

python train.py --bs $BS --comment _${METHOD}_${COMMENT}_dlr_${LR}_${MODEL}_ndf${NDF}_data_${LABELS}_${UNLABELS}_alekseitransform_samplingrandom_fold${FOLD_ID} \
    --kfold_split_file ${PKL_FILE} --n_epochs 500 --model_name ${MODEL} --root ${DATA_DIR}
