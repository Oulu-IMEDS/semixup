#!/usr/bin/env bash
if [ $# -lt 10 ]
  then
    echo "Not enough arguments. At least 10."
    exit 0
fi

FOLD_ID=$1
BS=$2

LABELS=$3
UNLABELS=$4
MODEL=$5
NDF=$6
N_BATCHES=$7
METHOD=$8
COMMENT=$9
ROOT=${10}
REMOVED_LOSSES=${11}

CDIR=$PWD

PKL_FILE=${CDIR}/scripts/processed_data/Metadata/cv_split_5fold_l_${LABELS}_u_${UNLABELS}_False_col_None.pkl

echo "Loading file $PKL_FILE"

LR=1e-4
DROP_RATE=0.35
N_EPOCHS=500

cd ${CDIR}/${METHOD}

echo "python ./train.py --bs $BS --comment ${METHOD}_${COMMENT}_dlr_${LR}_${MODEL}_ndf${NDF}_data_${LABELS}_${UNLABELS}_hoangtransform_samplingramdom_fold${FOLD_ID} \
    --drop_rate ${DROP_RATE} --n_features ${NDF} --model_name ${MODEL} \
    --lr ${LR} --n_labels ${LABELS} --n_epochs ${N_EPOCHS} \
    --n_unlabels ${UNLABELS} --unlabeled_target_column None --equal_unlabels --fold_index $FOLD_ID \
    --kfold_split_file $PKL_FILE --n_training_batches ${N_BATCHES} --root ${ROOT} --removed_losses ${REMOVED_LOSSES}"

python ./train.py --bs $BS --comment _${METHOD}_${COMMENT}_dlr_${LR}_${MODEL}_ndf${NDF}_data_${LABELS}_${UNLABELS}_hoangtransform_samplingrandom_fold${FOLD_ID} \
    --drop_rate ${DROP_RATE} --n_features ${NDF} --model_name ${MODEL} \
    --lr ${LR} --n_labels ${LABELS} --n_epochs ${N_EPOCHS} \
    --n_unlabels ${UNLABELS} --unlabeled_target_column None --equal_unlabels --fold_index $FOLD_ID \
    --kfold_split_file $PKL_FILE --n_training_batches ${N_BATCHES} --root ${ROOT} --removed_losses ${REMOVED_LOSSES}
