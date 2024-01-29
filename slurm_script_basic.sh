#!/bin/bash
#SBATCH --job-name=rpm-lds
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=1:00:0
#SBATCH --requeue
#SBATCH --exclude=gpu-sr670-20

echo "Launching a python run"
date

source /nfs/nhome/live/jheald/.bashrc

module load miniconda
conda deactivate
conda activate svae

export WANDB_API_KEY=9ae130eea17d49e2bd1deafd27c8a8de06f66830

cd /tmp
mkdir .mujoco
cp -R /nfs/nhome/live/jheald/.mujoco/mujoco210 /tmp/.mujoco/
export HOME=/tmp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/.mujoco/mujoco210/bin
export CPATH=$/nfs/nhome/live/jheald/.conda/envs/svae/include
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=0

python3 -u /nfs/nhome/live/jheald/svae/main.py --save_dir ${1} --use_delta_nat_q ${2} --use_delta_nat_f_tilde ${3} --f_time_dependent ${4} --use_ansatz ${5} --inference_method ${6} --jax_seed ${7}

rm -rf /tmp/.bashrc
rm -rf /tmp/.mujoco/

mv ${SLURM_SUBMIT_DIR}/slurm-${SLURM_JOB_ID}.out ${SLURM_SUBMIT_DIR}/runs/${1}/

# sbatch slurm_script_basic.sh '/nfs/nhome/live/jheald/svae/runs/NoDeltaQ_NoDeltaF_NoDeltaFNP_NoFTimeDepend_NoAnsatz_MyInference_svae_seed0' '0' '0' '0' '1' 'rpm' 0
