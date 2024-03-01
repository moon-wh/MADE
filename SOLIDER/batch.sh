#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=8G
#SBATCH -C gmem16
#SBATCH --job-name=PAR_MADE
#SBATCH --output=PAR_PRCC_output-MADE-1.txt
#SBATCH --gres-flags=enforce-binding

echo "*"{,,,,,,,,,}
echo $SLURM_JOB_ID
echo "*"{,,,,,,,,,}

nvidia-smi
source ~/.bashrc
cd /home/sriniana/projects/MADE/SOLIDER

CONDA_BASE=$(conda info --base) ;
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate solider
   
NUM_GPU=1
GPUS=0
PORT=12346

# CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port=$PORT train.py --cfg ./configs/peta_zs.yaml

CUDA_VISIBLE_DEVICES=$GPUS python demo_PETA_ccvid.py