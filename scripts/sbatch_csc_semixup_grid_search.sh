#!/bin/bash
#SBATCH --account=project_2002147
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1,nvme:3
#SBATCH --array=0-99

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.2.0

WEIGHTS=(0.25 0.5 1.0 2.0 4.0 8.0 16.0 32.0 64.0 128.0)
ID=${SLURM_ARRAY_TASK_ID}
let IO_MNF_ID=$ID/10
let IC_MNF_ID=$ID%10
IO_MNF_COEF=${WEIGHTS[IO_MNF_ID]}
IC_MNF_COEF=${WEIGHTS[IC_MNF_ID]}

COMMENT=weights.1.${IO_MNF_COEF}.${IC_MNF_COEF}

ARCHIVED_DATA=/scratch/project_2002147/hoang/data/MOST_OAI_00_0_2_cropped.tar.gz
DATA_DIR=${LOCAL_SCRATCH}/data
echo "mkdir -p ${DATA_DIR}"
mkdir -p ${DATA_DIR}
echo "Copying data to the node..."
echo "rsync ${ARCHIVED_DATA} ${DATA_DIR}"
rsync ${ARCHIVED_DATA} ${DATA_DIR}
echo "Done!"

echo "Extracting the data..."
tar -xzf ${DATA_DIR}/MOST_OAI_00_0_2_cropped.tar.gz -C ${DATA_DIR}
echo "Done!"

echo "Start the job..."
CDIR=$PWD

PKL_FILE=${CDIR}/scripts/processed_data/Metadata/cv_split_5fold_l_${LABELS}_u_${UNLABELS}_False_col_None.pkl

echo "Loading file $PKL_FILE"

FOLD_ID=1
NDF=32
BS=40
LR=1e-4
DROP_RATE=0.35
N_EPOCHS=300
SEED=12345
MODEL=cvgg2hv
METHOD=semixup
REMOVED_LOSSES=none
LABELS=500
UNLABELS=500
SKIP=1

let a=\(5*${UNLABELS}+${BS}-1\)
let b=$a/${BS}
let N_BATCHES=${SKIP}*$b

cd ${CDIR}/${METHOD}

python ./train.py --bs $BS --comment _${METHOD}_${COMMENT}_dlr_${LR}_${MODEL}_ndf${NDF}_data_${LABELS}_${UNLABELS}_hoangtransform_samplingrandom_fold${FOLD_ID} \
    --drop_rate ${DROP_RATE} --n_features ${NDF} --model_name ${MODEL} \
    --lr ${LR} --n_labels ${LABELS} --n_epochs ${N_EPOCHS} --seed ${SEED} --method_name ${METHOD} \
    --n_unlabels ${UNLABELS} --unlabeled_target_column None --equal_unlabels --fold_index $FOLD_ID \
    --kfold_split_file $PKL_FILE --n_training_batches ${N_BATCHES} --root ${DATA_DIR} --removed_losses ${REMOVED_LOSSES} \
    --in_mnf_coef $IO_MNF_COEF --in_mnf_coef $IO_MNF_COEF --ic_coef $IC_MNF_COEF
echo "Done the job!"
