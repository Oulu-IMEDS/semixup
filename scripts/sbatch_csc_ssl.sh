#!/bin/bash
#SBATCH --job-name=koa
#SBATCH --account=project_2002147
#SBATCH --partition=gpu
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpu:v100:1,nvme:3
#SBATCH --array=1-29

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.2.0

MODEL=$1
METHOD=$2
COMMENT=$3
SEED_ID=$4 # 0-5

SEED_LIST=(80122 66371 39333 67462 77665)
SEED=${SEED_LIST[SEED_ID]}

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
echo srun ./exp_ssl_settings_csc.sh 1 40 ${SLURM_ARRAY_TASK_ID} ${COMMENT} ${MODEL} ${METHOD} ${DATA_DIR}/MOST_OAI_00_0_2_cropped none ${SEED}
srun ./exp_ssl_settings_csc.sh 1 40 ${SLURM_ARRAY_TASK_ID} ${COMMENT} ${MODEL} ${METHOD} ${DATA_DIR}/MOST_OAI_00_0_2_cropped "none" ${SEED}
echo "Done the job!"