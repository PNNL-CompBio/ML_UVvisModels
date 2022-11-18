#!/bin/bash
#SBATCH -p dlt
#SBATCH -A mars_pli
#SBATCH -J UVSchNet
#SBATCH -t 4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -o OptSchNet-%A.out
#SBATCH -e OptSchNet-%A.err

module purge
module load gcc/7.1.0
module load cuda/11.1
module load python/anaconda3.2019.3

source activate /people/mcna892/.conda/envs/g4mp2_ml

python train_models_multi.py -d cuda

